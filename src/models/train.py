"""
Training script with MLflow experiment tracking.
"""

import os
import sys
import yaml
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
import mlflow
import mlflow.keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.baseline_cnn import create_and_compile_model
from src.data.preprocess import create_data_generators


def load_params(params_file: str = "params.yaml") -> dict:
    """Load parameters from YAML file."""
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)
    return params


def plot_training_history(history, output_dir: str = "models"):
    """
    Plot training history (loss and accuracy curves).

    Args:
        history: Keras History object
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Plot loss
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = output_path / 'training_history.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Training plots saved to: {plot_path}")
    return str(plot_path)


def evaluate_model(model, test_generator):
    """
    Evaluate model on test set and generate confusion matrix.

    Args:
        model: Trained Keras model
        test_generator: Test data generator

    Returns:
        Dictionary with evaluation metrics
    """
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_precision, test_recall, test_auc = model.evaluate(
        test_generator,
        verbose=1
    )

    # Calculate F1 score
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)

    metrics = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_auc': float(test_auc),
        'test_f1': float(test_f1)
    }

    print(f"\nTest Results:")
    print(f"  Loss:      {test_loss:.4f}")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  F1 Score:  {test_f1:.4f}")
    print(f"  AUC:       {test_auc:.4f}")

    return metrics


def train_model(
    data_dir: str = "data/processed",
    params_file: str = "params.yaml",
    output_dir: str = "models",
    experiment_name: str = "cats-vs-dogs-classification"
):
    """
    Train the model with MLflow tracking.

    Args:
        data_dir: Directory containing processed data
        params_file: Path to parameters YAML file
        output_dir: Directory to save model and artifacts
        experiment_name: MLflow experiment name
    """
    # Load parameters
    params = load_params(params_file)
    train_params = params['train']
    model_params = params['model']

    print("=" * 60)
    print("Training Cats vs Dogs Classifier")
    print("=" * 60)
    print(f"Experiment: {experiment_name}")
    print(f"Parameters: {train_params}")
    print("=" * 60)

    # Set MLflow tracking
    mlflow.set_experiment(experiment_name)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("learning_rate", train_params['learning_rate'])
        mlflow.log_param("batch_size", train_params['batch_size'])
        mlflow.log_param("epochs", train_params['epochs'])
        mlflow.log_param("img_size", train_params['img_size'])
        mlflow.log_param("dropout_rate", model_params['dropout_rate'])
        mlflow.log_param("architecture", model_params['architecture'])

        # Create data generators
        print("\nLoading data...")
        train_gen, val_gen, test_gen = create_data_generators(
            data_dir=data_dir,
            img_size=tuple(train_params['img_size']),
            batch_size=train_params['batch_size'],
            augment_train=params['data']['augmentation']
        )

        print(f"Train samples: {train_gen.samples}")
        print(f"Val samples:   {val_gen.samples}")
        print(f"Test samples:  {test_gen.samples}")

        # Create and compile model
        print("\nBuilding model...")
        model = create_and_compile_model(
            input_shape=tuple(model_params['input_shape']),
            dropout_rate=model_params['dropout_rate'],
            learning_rate=train_params['learning_rate']
        )

        model.summary()

        # Callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=str(output_path / 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]

        # Train model
        print("\nStarting training...")
        history = model.fit(
            train_gen,
            epochs=train_params['epochs'],
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )

        # Log metrics from training
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]

        mlflow.log_metric("final_train_accuracy", final_train_acc)
        mlflow.log_metric("final_val_accuracy", final_val_acc)
        mlflow.log_metric("final_train_loss", final_train_loss)
        mlflow.log_metric("final_val_loss", final_val_loss)

        # Plot and log training history
        plot_path = plot_training_history(history, output_dir=output_dir)
        mlflow.log_artifact(plot_path)

        # Evaluate on test set
        test_metrics = evaluate_model(model, test_gen)

        # Log test metrics
        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Save final model
        model_path = output_path / 'cats_dogs_classifier.h5'
        model.save(model_path)
        print(f"\nModel saved to: {model_path}")

        # Log model to MLflow
        mlflow.keras.log_model(model, "model")

        # Save metrics to JSON
        all_metrics = {
            "final_train_accuracy": final_train_acc,
            "final_val_accuracy": final_val_acc,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            **test_metrics,
            "timestamp": datetime.now().isoformat()
        }

        metrics_path = output_path / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        mlflow.log_artifact(str(metrics_path))

        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        print("=" * 60)

        return model, history, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Cats vs Dogs classifier')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--params_file', type=str, default='params.yaml',
                        help='Path to parameters YAML file')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Path to save trained model')
    parser.add_argument('--experiment_name', type=str,
                        default='cats-vs-dogs-classification',
                        help='MLflow experiment name')

    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        params_file=args.params_file,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name
    )
