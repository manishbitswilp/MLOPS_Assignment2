"""
Unit tests for data preprocessing functions.
"""

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
import tempfile
from PIL import Image

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocess import (
    preprocess_single_image,
    preprocess_image_bytes
)


class TestImagePreprocessing:
    """Test cases for image preprocessing functions."""

    @pytest.fixture
    def sample_image_path(self):
        """Create a temporary test image."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            # Create a simple RGB image
            img = Image.new('RGB', (300, 300), color='red')
            img.save(tmp.name)
            yield tmp.name
            # Cleanup
            Path(tmp.name).unlink()

    @pytest.fixture
    def sample_image_bytes(self):
        """Create sample image bytes."""
        # Create a simple RGB image
        img = Image.new('RGB', (300, 300), color='blue')
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img.save(tmp.name)
            with open(tmp.name, 'rb') as f:
                image_bytes = f.read()
            Path(tmp.name).unlink()
            return image_bytes

    def test_preprocess_single_image_shape(self, sample_image_path):
        """Test that preprocessed image has correct shape."""
        img_size = (224, 224)
        result = preprocess_single_image(sample_image_path, img_size)

        assert result.shape == (1, 224, 224, 3), \
            f"Expected shape (1, 224, 224, 3), got {result.shape}"

    def test_preprocess_single_image_normalization(self, sample_image_path):
        """Test that pixel values are normalized to [0, 1] range."""
        result = preprocess_single_image(sample_image_path, img_size=(224, 224))

        assert result.min() >= 0.0, f"Min value {result.min()} is less than 0"
        assert result.max() <= 1.0, f"Max value {result.max()} is greater than 1"
        assert result.dtype == np.float32 or result.dtype == np.float64

    def test_preprocess_single_image_custom_size(self, sample_image_path):
        """Test preprocessing with custom image size."""
        img_size = (128, 128)
        result = preprocess_single_image(sample_image_path, img_size)

        assert result.shape == (1, 128, 128, 3), \
            f"Expected shape (1, 128, 128, 3), got {result.shape}"

    def test_preprocess_image_bytes_shape(self, sample_image_bytes):
        """Test that preprocessing from bytes produces correct shape."""
        img_size = (224, 224)
        result = preprocess_image_bytes(sample_image_bytes, img_size)

        assert result.shape == (1, 224, 224, 3), \
            f"Expected shape (1, 224, 224, 3), got {result.shape}"

    def test_preprocess_image_bytes_normalization(self, sample_image_bytes):
        """Test that pixel values from bytes are normalized correctly."""
        result = preprocess_image_bytes(sample_image_bytes, img_size=(224, 224))

        assert result.min() >= 0.0, f"Min value {result.min()} is less than 0"
        assert result.max() <= 1.0, f"Max value {result.max()} is greater than 1"

    def test_preprocess_nonexistent_file(self):
        """Test that preprocessing raises error for nonexistent file."""
        with pytest.raises(Exception):
            preprocess_single_image("nonexistent_file.jpg")

    def test_preprocess_invalid_bytes(self):
        """Test that preprocessing raises error for invalid image bytes."""
        invalid_bytes = b"not an image"
        with pytest.raises(Exception):
            preprocess_image_bytes(invalid_bytes)

    def test_preprocess_batch_dimension(self, sample_image_path):
        """Test that batch dimension is added correctly."""
        result = preprocess_single_image(sample_image_path)

        assert len(result.shape) == 4, \
            f"Expected 4 dimensions, got {len(result.shape)}"
        assert result.shape[0] == 1, \
            f"Expected batch size 1, got {result.shape[0]}"

    def test_preprocess_rgb_channels(self, sample_image_path):
        """Test that image has 3 RGB channels."""
        result = preprocess_single_image(sample_image_path)

        assert result.shape[-1] == 3, \
            f"Expected 3 RGB channels, got {result.shape[-1]}"


class TestDataAugmentation:
    """Test cases for data augmentation (indirect testing through generators)."""

    def test_augmentation_preserves_shape(self):
        """Test that augmentation preserves image shape."""
        # Create a simple test image
        original = np.random.rand(1, 224, 224, 3).astype(np.float32)

        # Apply basic transformations manually
        from tensorflow.keras.preprocessing.image import apply_affine_transform

        transformed = apply_affine_transform(
            original[0],
            theta=10,  # rotation
            tx=0.1,    # horizontal shift
            ty=0.1     # vertical shift
        )

        assert transformed.shape == (224, 224, 3), \
            f"Expected shape (224, 224, 3), got {transformed.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
