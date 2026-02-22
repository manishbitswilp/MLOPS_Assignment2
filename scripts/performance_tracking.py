"""
Post-deployment model performance tracking.

This script evaluates the deployed model's performance by:
1. Loading test dataset with ground truth labels
2. Sending images to the deployed API
3. Collecting predictions
4. Calculating performance metrics
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import requests
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_test_images(test_dir: str, max_samples: int = None) -> Tuple[List[str], List[int]]:
    """
    Load test image paths and labels.

    Args:
        test_dir: Directory containing test images (with cat/ and dog/ subdirs)
        max_samples: Maximum number of samples to test (None for all)

    Returns:
        Tuple of (image_paths, labels) where labels are 0=cat, 1=dog
    """
    test_path = Path(test_dir)

    image_paths = []
    labels = []

    # Load cat images (label=0)
    cat_dir = test_path / "cat"
    if cat_dir.exists():
        cat_images = list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.png"))
        image_paths.extend([str(p) for p in cat_images])
        labels.extend([0] * len(cat_images))

    # Load dog images (label=1)
    dog_dir = test_path / "dog"
    if dog_dir.exists():
        dog_images = list(dog_dir.glob("*.jpg")) + list(dog_dir.glob("*.png"))
        image_paths.extend([str(p) for p in dog_images])
        labels.extend([1] * len(dog_images))

    print(f"Loaded {len(image_paths)} test images ({labels.count(0)} cats, {labels.count(1)} dogs)")

    # Limit samples if requested
    if max_samples and len(image_paths) > max_samples:
        indices = np.random.choice(len(image_paths), max_samples, replace=False)
        image_paths = [image_paths[i] for i in indices]
        labels = [labels[i] for i in indices]
        print(f"Limited to {max_samples} samples")

    return image_paths, labels


def predict_image(api_url: str, image_path: str, timeout: int = 30) -> Dict:
    """
    Send image to API and get prediction.

    Args:
        api_url: API endpoint URL (e.g., http://localhost:8000/predict)
        image_path: Path to image file
        timeout: Request timeout in seconds

    Returns:
        Prediction result dictionary
    """
    with open(image_path, 'rb') as f:
        files = {'file': (Path(image_path).name, f, 'image/jpeg')}
        response = requests.post(api_url, files=files, timeout=timeout)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API request failed: {response.status_code} - {response.text}")


def evaluate_predictions(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str] = ['cat', 'dog']
) -> Dict:
    """
    Calculate evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names for reporting

    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'f1_score': float(f1_score(y_true, y_pred)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    metrics['classification_report'] = report

    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str
):
    """
    Plot and save confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: Class names
        output_path: Path to save plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - Post-Deployment Evaluation')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def run_performance_tracking(
    api_url: str,
    test_dir: str,
    output_dir: str = "models/performance",
    max_samples: int = None
) -> Dict:
    """
    Run complete performance tracking pipeline.

    Args:
        api_url: Base URL of the deployed API
        test_dir: Directory containing test images
        output_dir: Directory to save results
        max_samples: Maximum number of samples to test

    Returns:
        Dictionary with performance metrics
    """
    print("=" * 70)
    print("POST-DEPLOYMENT PERFORMANCE TRACKING")
    print("=" * 70)
    print(f"API URL: {api_url}")
    print(f"Test directory: {test_dir}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load test images
    print("\n" + "-" * 70)
    print("Loading test images...")
    print("-" * 70)
    image_paths, y_true = load_test_images(test_dir, max_samples)

    if len(image_paths) == 0:
        print("❌ No test images found!")
        return {}

    # Make predictions
    print("\n" + "-" * 70)
    print("Making predictions via API...")
    print("-" * 70)

    y_pred = []
    confidences = []
    latencies = []

    predict_url = f"{api_url}/predict"

    for i, image_path in enumerate(image_paths):
        try:
            start_time = datetime.now()
            result = predict_image(predict_url, image_path)
            latency = (datetime.now() - start_time).total_seconds()

            # Convert class to label (cat=0, dog=1)
            pred_label = 1 if result['class'] == 'dog' else 0
            y_pred.append(pred_label)
            confidences.append(result['confidence'])
            latencies.append(latency)

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(image_paths)} images...")

        except Exception as e:
            print(f"Error predicting {image_path}: {str(e)}")
            continue

    print(f"Successfully predicted {len(y_pred)}/{len(image_paths)} images")

    # Calculate metrics
    print("\n" + "-" * 70)
    print("Calculating metrics...")
    print("-" * 70)

    metrics = evaluate_predictions(y_true[:len(y_pred)], y_pred)

    # Add latency metrics
    metrics['avg_latency_seconds'] = float(np.mean(latencies))
    metrics['median_latency_seconds'] = float(np.median(latencies))
    metrics['p95_latency_seconds'] = float(np.percentile(latencies, 95))
    metrics['avg_confidence'] = float(np.mean(confidences))

    # Print results
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"\nAvg Latency:    {metrics['avg_latency_seconds']:.3f}s")
    print(f"Median Latency: {metrics['median_latency_seconds']:.3f}s")
    print(f"P95 Latency:    {metrics['p95_latency_seconds']:.3f}s")
    print(f"Avg Confidence: {metrics['avg_confidence']:.4f}")

    # Plot confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    plot_path = output_path / f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plot_confusion_matrix(cm, ['cat', 'dog'], str(plot_path))

    # Save metrics to JSON
    metrics['timestamp'] = datetime.now().isoformat()
    metrics['api_url'] = api_url
    metrics['num_samples'] = len(y_pred)

    metrics_path = output_path / f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to: {metrics_path}")

    print("\n" + "=" * 70)
    print("Performance tracking completed!")
    print("=" * 70)

    return metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Track post-deployment model performance'
    )
    parser.add_argument(
        '--api-url',
        type=str,
        default='http://localhost:8000',
        help='Base URL of the deployed API'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        default='data/processed/test',
        help='Directory containing test images'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/performance',
        help='Directory to save results'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to test (default: all)'
    )

    args = parser.parse_args()

    try:
        metrics = run_performance_tracking(
            api_url=args.api_url,
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            max_samples=args.max_samples
        )

        if metrics:
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
