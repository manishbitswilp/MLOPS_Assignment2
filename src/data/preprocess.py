"""
Image preprocessing and data augmentation for Cats vs Dogs classification.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json


def create_data_generators(
    data_dir: str = "data/processed",
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    augment_train: bool = True
) -> Tuple[tf.keras.preprocessing.image.DirectoryIterator,
           tf.keras.preprocessing.image.DirectoryIterator,
           tf.keras.preprocessing.image.DirectoryIterator]:
    """
    Create data generators for train, validation, and test sets.

    Args:
        data_dir: Root directory containing train/val/test subdirectories
        img_size: Target image size (height, width)
        batch_size: Batch size for training
        augment_train: Whether to apply data augmentation to training set

    Returns:
        Tuple of (train_generator, val_generator, test_generator)
    """
    data_path = Path(data_dir)

    # Training data generator with augmentation
    if augment_train:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,            # Normalize to [0, 1]
            rotation_range=20,               # Random rotation ±20 degrees
            width_shift_range=0.2,           # Random horizontal shift
            height_shift_range=0.2,          # Random vertical shift
            shear_range=0.2,                 # Shear transformation
            zoom_range=0.2,                  # Random zoom
            horizontal_flip=True,            # Random horizontal flip
            brightness_range=[0.8, 1.2],     # Random brightness adjustment
            fill_mode='nearest'              # Fill strategy for new pixels
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # Validation and test generators (no augmentation, only normalization)
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # Create generators
    train_generator = train_datagen.flow_from_directory(
        data_path / "train",
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',              # Binary classification (cat=0, dog=1)
        shuffle=True,
        seed=42
    )

    val_generator = val_test_datagen.flow_from_directory(
        data_path / "val",
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_directory(
        data_path / "test",
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    return train_generator, val_generator, test_generator


def preprocess_single_image(
    image_path: str,
    img_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Preprocess a single image for inference.

    Args:
        image_path: Path to the image file
        img_size: Target image size (height, width)

    Returns:
        Preprocessed image array with shape (1, height, width, 3)
    """
    img = tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=img_size
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    return img_array


def preprocess_image_bytes(
    image_bytes: bytes,
    img_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Preprocess image from bytes (for API inference).

    Args:
        image_bytes: Raw image bytes
        img_size: Target image size (height, width)

    Returns:
        Preprocessed image array with shape (1, height, width, 3)
    """
    # Decode image from bytes
    img = tf.image.decode_image(image_bytes, channels=3)
    img = tf.image.resize(img, img_size)
    img = img / 255.0  # Normalize to [0, 1]
    img = tf.expand_dims(img, axis=0)  # Add batch dimension

    return img.numpy()


def save_preprocessing_metadata(
    output_path: str,
    img_size: Tuple[int, int],
    batch_size: int,
    class_names: list
) -> None:
    """
    Save preprocessing metadata for reproducibility.

    Args:
        output_path: Path to save metadata JSON file
        img_size: Image size used for preprocessing
        batch_size: Batch size used for training
        class_names: List of class names
    """
    metadata = {
        "img_height": img_size[0],
        "img_width": img_size[1],
        "batch_size": batch_size,
        "class_names": class_names,
        "normalization": "rescale_1_255",
        "color_mode": "rgb"
    }

    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Preprocessing metadata saved to: {output_path}")


if __name__ == "__main__":
    print("=" * 50)
    print("Testing Data Generators")
    print("=" * 50)

    # Create generators
    train_gen, val_gen, test_gen = create_data_generators(
        data_dir="data/processed",
        img_size=(224, 224),
        batch_size=32,
        augment_train=True
    )

    print(f"\nTrain samples: {train_gen.samples}")
    print(f"Val samples:   {val_gen.samples}")
    print(f"Test samples:  {test_gen.samples}")

    print(f"\nClass indices: {train_gen.class_indices}")

    # Save metadata
    save_preprocessing_metadata(
        output_path="data/preprocessing_metadata.json",
        img_size=(224, 224),
        batch_size=32,
        class_names=list(train_gen.class_indices.keys())
    )

    # Test a batch
    print("\nTesting batch retrieval...")
    x_batch, y_batch = next(train_gen)
    print(f"Batch shape: {x_batch.shape}")
    print(f"Labels shape: {y_batch.shape}")
    print(f"Pixel value range: [{x_batch.min():.3f}, {x_batch.max():.3f}]")

    print("\nPreprocessing pipeline ready!")
