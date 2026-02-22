"""
Utility functions for model inference.
"""

import json
import numpy as np
import h5py
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

    # Keras 3 layer kwargs unknown to Keras 2
    _UNKNOWN_LAYER_KWARGS = {'quantization_config', 'lora_rank'}

    def _patch_config(cfg):
        """Recursively convert Keras 3 config idioms to Keras 2 equivalents."""
        if isinstance(cfg, list):
            return [_patch_config(v) for v in cfg]
        if not isinstance(cfg, dict):
            return cfg

        # DTypePolicy objects -> plain dtype string (e.g. 'float32')
        if cfg.get('class_name') == 'DTypePolicy':
            return cfg.get('config', {}).get('name', 'float32')

        # Keras 3 serialized objects have 'module' and 'registered_name';
        # Keras 2 only uses 'class_name' and 'config' — strip the extras.
        if 'class_name' in cfg and 'module' in cfg:
            cfg = {'class_name': cfg['class_name'], 'config': cfg.get('config', {})}

        # InputLayer: 'batch_shape' -> 'batch_input_shape', drop 'optional'
        if cfg.get('class_name') == 'InputLayer':
            inner = dict(cfg.get('config', {}))
            inner.pop('optional', None)
            if 'batch_shape' in inner:
                inner['batch_input_shape'] = inner.pop('batch_shape')
            cfg = dict(cfg)
            cfg['config'] = _patch_config(inner)
            return cfg

        patched = {k: _patch_config(v) for k, v in cfg.items()}

        # Drop Keras 3 only layer kwargs from the inner 'config' dict
        if 'config' in patched and isinstance(patched['config'], dict):
            for key in _UNKNOWN_LAYER_KWARGS:
                patched['config'].pop(key, None)

        return patched

    with h5py.File(model_path, 'r') as f:
        raw_config = f.attrs['model_config']
        model_config = json.loads(raw_config)

    patched_config = _patch_config(model_config)
    model = tf.keras.models.model_from_json(json.dumps(patched_config))
    model.load_weights(model_path)

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
