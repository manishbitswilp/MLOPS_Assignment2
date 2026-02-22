"""
Unit tests for inference utilities.
"""

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.inference import (
    load_model,
    preprocess_image,
    predict,
    predict_from_bytes
)


class TestModelLoading:
    """Test cases for model loading."""

    def test_load_nonexistent_model(self):
        """Test that loading nonexistent model raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent_model.h5")

    def test_load_model_success(self):
        """Test successful model loading."""
        import json
        import h5py as real_h5py

        mock_model = MagicMock()
        fake_config = json.dumps({'class_name': 'Sequential', 'config': {'layers': []}})

        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.attrs = {'model_config': fake_config}

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with patch.object(real_h5py, 'File', return_value=mock_file), \
                 patch.object(tf.keras.models, 'model_from_json', return_value=mock_model):
                model = load_model(tmp_path)
                assert model is not None
        finally:
            Path(tmp_path).unlink()


class TestImagePreprocessing:
    """Test cases for image preprocessing in inference."""

    def test_preprocess_image_shape(self):
        """Test that preprocessing produces correct shape."""
        # Create fake image bytes
        img = tf.random.uniform((300, 300, 3), minval=0, maxval=255, dtype=tf.float32)
        img_bytes = tf.image.encode_jpeg(tf.cast(img, tf.uint8)).numpy()

        result = preprocess_image(img_bytes, img_size=(224, 224))

        assert result.shape == (1, 224, 224, 3), \
            f"Expected shape (1, 224, 224, 3), got {result.shape}"

    def test_preprocess_image_normalization(self):
        """Test that image is normalized to [0, 1]."""
        # Create fake image bytes
        img = tf.random.uniform((300, 300, 3), minval=0, maxval=255, dtype=tf.float32)
        img_bytes = tf.image.encode_jpeg(tf.cast(img, tf.uint8)).numpy()

        result = preprocess_image(img_bytes, img_size=(224, 224))

        assert result.min() >= 0.0, f"Min value {result.min()} is less than 0"
        assert result.max() <= 1.0, f"Max value {result.max()} is greater than 1"

    def test_preprocess_image_invalid_bytes(self):
        """Test that invalid bytes raise an error."""
        invalid_bytes = b"not an image"

        with pytest.raises(Exception):
            preprocess_image(invalid_bytes)


class TestPrediction:
    """Test cases for prediction functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        # Mock predict method to return a prediction
        model.predict.return_value = np.array([[0.85]])
        return model

    @pytest.fixture
    def sample_image(self):
        """Create a sample preprocessed image."""
        return np.random.rand(1, 224, 224, 3).astype(np.float32)

    def test_predict_dog_class(self, mock_model, sample_image):
        """Test prediction for dog class (probability > 0.5)."""
        mock_model.predict.return_value = np.array([[0.85]])

        result = predict(mock_model, sample_image)

        assert result['class'] == 'dog'
        assert result['confidence'] == 0.85
        assert result['dog_probability'] == 0.85
        assert result['cat_probability'] == 0.15

    def test_predict_cat_class(self, mock_model, sample_image):
        """Test prediction for cat class (probability < 0.5)."""
        mock_model.predict.return_value = np.array([[0.25]])

        result = predict(mock_model, sample_image)

        assert result['class'] == 'cat'
        assert result['confidence'] == 0.75
        assert result['dog_probability'] == 0.25
        assert result['cat_probability'] == 0.75

    def test_predict_output_format(self, mock_model, sample_image):
        """Test that prediction output has correct format."""
        result = predict(mock_model, sample_image)

        assert 'class' in result
        assert 'confidence' in result
        assert 'dog_probability' in result
        assert 'cat_probability' in result

        assert isinstance(result['class'], str)
        assert isinstance(result['confidence'], float)
        assert isinstance(result['dog_probability'], float)
        assert isinstance(result['cat_probability'], float)

    def test_predict_probability_range(self, mock_model, sample_image):
        """Test that probabilities are in valid range [0, 1]."""
        result = predict(mock_model, sample_image)

        assert 0.0 <= result['confidence'] <= 1.0
        assert 0.0 <= result['dog_probability'] <= 1.0
        assert 0.0 <= result['cat_probability'] <= 1.0

    def test_predict_probabilities_sum_to_one(self, mock_model, sample_image):
        """Test that cat and dog probabilities sum to approximately 1."""
        result = predict(mock_model, sample_image)

        prob_sum = result['dog_probability'] + result['cat_probability']
        assert abs(prob_sum - 1.0) < 0.01, \
            f"Probabilities sum to {prob_sum}, expected ~1.0"

    def test_predict_boundary_case_05(self, mock_model, sample_image):
        """Test prediction at boundary (0.5)."""
        mock_model.predict.return_value = np.array([[0.5]])

        result = predict(mock_model, sample_image)

        # At exactly 0.5, should predict cat (as condition is > 0.5 for dog)
        assert result['class'] == 'cat'
        assert result['confidence'] == 0.5

    def test_predict_extreme_confidence(self, mock_model, sample_image):
        """Test prediction with very high confidence."""
        mock_model.predict.return_value = np.array([[0.99]])

        result = predict(mock_model, sample_image)

        assert result['class'] == 'dog'
        assert result['confidence'] >= 0.99
        assert result['dog_probability'] >= 0.99

    @patch('src.utils.inference.preprocess_image')
    def test_predict_from_bytes(self, mock_preprocess, mock_model):
        """Test end-to-end prediction from bytes."""
        # Mock preprocessing to return a sample image
        mock_preprocess.return_value = np.random.rand(1, 224, 224, 3)

        # Mock model prediction
        mock_model.predict.return_value = np.array([[0.75]])

        # Create fake image bytes
        fake_bytes = b"fake image bytes"

        result = predict_from_bytes(mock_model, fake_bytes, img_size=(224, 224))

        assert 'class' in result
        assert 'confidence' in result
        mock_preprocess.assert_called_once_with(fake_bytes, (224, 224))


class TestRounding:
    """Test cases for output rounding."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.predict.return_value = np.array([[0.876543]])
        return model

    def test_confidence_rounding(self, mock_model):
        """Test that confidence is rounded to 4 decimal places."""
        sample_image = np.random.rand(1, 224, 224, 3)
        result = predict(mock_model, sample_image)

        # Check that all probabilities are rounded to 4 decimals
        assert len(str(result['confidence']).split('.')[-1]) <= 4
        assert len(str(result['dog_probability']).split('.')[-1]) <= 4
        assert len(str(result['cat_probability']).split('.')[-1]) <= 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
