"""
Baseline CNN model for Cats vs Dogs binary classification.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple


def create_baseline_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.5
) -> tf.keras.Model:
    """
    Create a baseline CNN model for binary image classification.

    Architecture:
        - Conv2D(32) -> MaxPooling2D
        - Conv2D(64) -> MaxPooling2D
        - Conv2D(128) -> MaxPooling2D
        - Flatten
        - Dense(128) -> Dropout
        - Dense(1, sigmoid)

    Args:
        input_shape: Shape of input images (height, width, channels)
        dropout_rate: Dropout rate for regularization (default: 0.5)

    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, name='conv1'),
        layers.MaxPooling2D((2, 2), name='pool1'),

        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
        layers.MaxPooling2D((2, 2), name='pool2'),

        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', name='conv3'),
        layers.MaxPooling2D((2, 2), name='pool3'),

        # Flatten and dense layers
        layers.Flatten(name='flatten'),
        layers.Dense(128, activation='relu', name='dense1'),
        layers.Dropout(dropout_rate, name='dropout'),

        # Output layer (binary classification)
        layers.Dense(1, activation='sigmoid', name='output')
    ], name='baseline_cnn')

    return model


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """
    Compile the model with optimizer, loss, and metrics.

    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer

    Returns:
        Compiled model
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    return model


def create_and_compile_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.5,
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """
    Create and compile the baseline CNN model.

    Args:
        input_shape: Shape of input images
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer

    Returns:
        Compiled Keras model ready for training
    """
    model = create_baseline_cnn(input_shape, dropout_rate)
    model = compile_model(model, learning_rate)

    return model


if __name__ == "__main__":
    # Test model creation
    print("=" * 50)
    print("Testing Baseline CNN Model")
    print("=" * 50)

    model = create_and_compile_model(
        input_shape=(224, 224, 3),
        dropout_rate=0.5,
        learning_rate=0.001
    )

    print("\nModel Summary:")
    model.summary()

    print(f"\nTotal parameters: {model.count_params():,}")
    print("\nModel created successfully!")
