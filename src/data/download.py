"""
Download and organize the Kaggle Cats and Dogs dataset.

This script downloads the dataset and organizes it into train/validation/test splits.
"""

import os
import zipfile
import shutil
from pathlib import Path
import random
from typing import List, Tuple


def download_dataset(data_dir: str = "data/raw") -> None:
    """
    Download the Kaggle Cats and Dogs dataset.

    Note: This requires kaggle CLI to be configured with API credentials.
    Install kaggle: pip install kaggle
    Configure: Place kaggle.json in ~/.kaggle/

    Args:
        data_dir: Directory to store the raw dataset
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    print("Downloading Kaggle Cats and Dogs dataset...")
    print("Note: Requires kaggle CLI configured with API credentials")

    # Download using Kaggle API
    os.system(f"kaggle datasets download -d salader/dogs-vs-cats -p {data_dir}")

    # Extract the zip file
    zip_path = data_path / "dogs-vs-cats.zip"
    if zip_path.exists():
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        print("Extraction complete!")

        # Clean up zip file
        zip_path.unlink()
    else:
        print(f"Warning: Expected zip file not found at {zip_path}")
        print("Please download manually from: https://www.kaggle.com/datasets/salader/dogs-vs-cats")


def organize_dataset(
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> None:
    """
    Organize dataset into train/validation/test splits.

    Args:
        raw_dir: Directory containing raw downloaded data
        output_dir: Directory to store organized splits
        train_ratio: Fraction of data for training (default: 0.8)
        val_ratio: Fraction of data for validation (default: 0.1)
        test_ratio: Fraction of data for testing (default: 0.1)
        seed: Random seed for reproducibility
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Train, val, and test ratios must sum to 1.0"

    random.seed(seed)

    raw_path = Path(raw_dir)
    output_path = Path(output_dir)

    # Find the training directory (might be nested)
    train_dir = None
    for subdir in raw_path.rglob("train"):
        if subdir.is_dir():
            train_dir = subdir
            break

    if train_dir is None:
        print(f"Error: Could not find 'train' directory in {raw_dir}")
        print("Please ensure the dataset is properly extracted")
        return

    print(f"Found training data in: {train_dir}")

    # Get all image files
    image_files = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))
    print(f"Found {len(image_files)} images")

    # Separate cats and dogs
    cat_files = [f for f in image_files if "cat" in f.name.lower()]
    dog_files = [f for f in image_files if "dog" in f.name.lower()]

    print(f"Cats: {len(cat_files)}, Dogs: {len(dog_files)}")

    # Shuffle
    random.shuffle(cat_files)
    random.shuffle(dog_files)

    # Split each class
    def split_files(files: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train = files[:n_train]
        val = files[n_train:n_train + n_val]
        test = files[n_train + n_val:]

        return train, val, test

    cat_train, cat_val, cat_test = split_files(cat_files)
    dog_train, dog_val, dog_test = split_files(dog_files)

    print(f"\nSplit distribution:")
    print(f"Train: {len(cat_train)} cats, {len(dog_train)} dogs")
    print(f"Val:   {len(cat_val)} cats, {len(dog_val)} dogs")
    print(f"Test:  {len(cat_test)} cats, {len(dog_test)} dogs")

    # Create directory structure
    splits = {
        "train": {"cat": cat_train, "dog": dog_train},
        "val": {"cat": cat_val, "dog": dog_val},
        "test": {"cat": cat_test, "dog": dog_test}
    }

    for split_name, classes in splits.items():
        for class_name, files in classes.items():
            target_dir = output_path / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)

            # Copy files
            for file_path in files:
                target_path = target_dir / file_path.name
                shutil.copy2(file_path, target_path)

    print(f"\nDataset organized in: {output_path}")
    print("Structure:")
    print("  train/")
    print("    cat/")
    print("    dog/")
    print("  val/")
    print("    cat/")
    print("    dog/")
    print("  test/")
    print("    cat/")
    print("    dog/")


if __name__ == "__main__":
    # Download dataset (uncomment if needed)
    # download_dataset()

    # Organize dataset
    print("=" * 50)
    print("Organizing Cats vs Dogs Dataset")
    print("=" * 50)
    organize_dataset()
    print("\nDone!")
