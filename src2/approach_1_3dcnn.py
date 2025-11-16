"""
3D CNN for WLASL Video Sign Language Recognition
"""

import os
import glob
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout

DATASET_DIR = "data/WLASL/start_kit/dataset"
NUM_FRAMES = 16
IMG_SIZE = (112, 112)

# --- Utils ---
def sample_frames(video_path, num_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        if i in indices:
            frame = cv2.resize(frame, IMG_SIZE)
            frames.append(frame/255.0)
    cap.release()
    return np.array(frames)

def load_dataset():
    X, y = [], []
    classes = sorted(os.listdir(DATASET_DIR))
    class_map = {cls:i for i, cls in enumerate(classes)}
    for cls in classes:
        files = glob.glob(os.path.join(DATASET_DIR, cls, "*.mp4"))
        for f in files:
            frames = sample_frames(f)
            if frames.shape[0] == NUM_FRAMES:
                X.append(frames)
                y.append(class_map[cls])
    X = np.array(X)
    y = tf.keras.utils.to_categorical(y, num_classes=len(classes))
    return X, y, classes

# --- Model ---
def build_3dcnn(num_frames=NUM_FRAMES, num_classes=100):
    model = Sequential([
        Conv3D(32, (3,3,3), activation='relu', input_shape=(num_frames, *IMG_SIZE,3)),
        MaxPooling3D((2,2,2)),
        Conv3D(64, (3,3,3), activation='relu'),
        MaxPooling3D((2,2,2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# --- Main ---
if __name__ == "__main__":
    X, y, classes = load_dataset()
    print(f"Loaded {X.shape[0]} videos, {len(classes)} classes")
    
    model = build_3dcnn(num_classes=len(classes))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, batch_size=4, epochs=10, validation_split=0.1)
