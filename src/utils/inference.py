"""
Utility functions for model inference.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Dict


def load_model(model_path: str) -> tf.keras.Model:
    """
    Load a trained Keras model from file.

    Args:
        model_path: Path to the saved model (.h5 file)

    Returns:
        Loaded Keras model
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from: {model_path}")

    return model


def preprocess_image(image_bytes: bytes, img_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess image bytes for inference.

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


def predict(model: tf.keras.Model, image: np.ndarray) -> Dict[str, float]:
    """
    Make prediction on a preprocessed image.

    Args:
        model: Trained Keras model
        image: Preprocessed image array with shape (1, height, width, 3)

    Returns:
        Dictionary with prediction results:
            - class: 'cat' or 'dog'
            - confidence: probability of predicted class (0-1)
            - dog_probability: probability of dog class
            - cat_probability: probability of cat class
    """
    # Make prediction
    prediction = model.predict(image, verbose=0)
    dog_probability = float(prediction[0][0])
    cat_probability = 1.0 - dog_probability

    # Determine class (dog=1, cat=0)
    predicted_class = 'dog' if dog_probability > 0.5 else 'cat'
    confidence = dog_probability if predicted_class == 'dog' else cat_probability

    return {
        'class': predicted_class,
        'confidence': round(confidence, 4),
        'dog_probability': round(dog_probability, 4),
        'cat_probability': round(cat_probability, 4)
    }


def predict_from_bytes(
    model: tf.keras.Model,
    image_bytes: bytes,
    img_size: Tuple[int, int] = (224, 224)
) -> Dict[str, float]:
    """
    End-to-end prediction from image bytes.

    Args:
        model: Trained Keras model
        image_bytes: Raw image bytes
        img_size: Target image size (height, width)

    Returns:
        Dictionary with prediction results
    """
    # Preprocess image
    image = preprocess_image(image_bytes, img_size)

    # Make prediction
    result = predict(model, image)

    return result


if __name__ == "__main__":
    # Test inference utilities
    print("=" * 50)
    print("Testing Inference Utilities")
    print("=" * 50)

    # This is a placeholder test - actual testing requires a trained model
    print("\nInference utilities loaded successfully!")
    print("Functions available:")
    print("  - load_model(model_path)")
    print("  - preprocess_image(image_bytes, img_size)")
    print("  - predict(model, image)")
    print("  - predict_from_bytes(model, image_bytes, img_size)")
