"""
model_cnn.py
------------
Defines the CNN model for static ASL sign recognition.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def build_cnn_model(input_shape=(64, 64, 3), num_classes=29):
    """Creates and returns a CNN model for ASL alphabet recognition."""
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Flatten(),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model
