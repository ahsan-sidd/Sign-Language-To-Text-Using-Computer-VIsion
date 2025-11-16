"""
Robust CNN + LSTM for WLASL Video Sign Language Recognition
Handles corrupted videos and variable-length clips
"""

import os
import glob
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt

def plot_training_curves(history, save_path="models/training_curves.png"):
    """
    Plots and saves training & validation loss and accuracy curves.
    """
    # Loss
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# ---------------- Configuration ----------------
DATASET_DIR = "data/WLASL/start_kit/dataset"
NUM_FRAMES = 16           # number of frames per video
IMG_SIZE = (224, 224)     # MobileNetV2 expects 224x224
BATCH_SIZE = 4
EPOCHS = 10

# ---------------- Utilities ----------------
def sample_frames(video_path, num_frames=NUM_FRAMES):
    """
    Sample 'num_frames' frames uniformly from video.
    Skip corrupted frames. Pad with last frame if video is too short.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError("Video has zero frames")
        
        # Sample frame indices
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                continue  # skip corrupted frame
            if i in indices:
                frame = cv2.resize(frame, IMG_SIZE)
                frames.append(frame / 255.0)
        cap.release()
        
        # Pad with last frame if fewer frames collected
        while len(frames) < num_frames:
            frames.append(frames[-1])
        frames = np.array(frames)
        return frames
    except Exception as e:
        print(f"[Warning] Skipping video {video_path}: {e}")
        return None

# ---------------- Load CNN Features ----------------
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(*IMG_SIZE,3))
feature_extractor = Model(base_model.input, base_model.output)

def video_to_features(video_path):
    frames = sample_frames(video_path)
    print(video_path)
    if frames is None:
        pass
        #return None
    features = feature_extractor.predict(frames, verbose=0)
    return features  # shape: (NUM_FRAMES, feature_dim)

def load_dataset():
    X, y = [], []
    classes = sorted(os.listdir(DATASET_DIR))
    class_map = {cls:i for i, cls in enumerate(classes)}
    
    for cls in classes:
        files = glob.glob(os.path.join(DATASET_DIR, cls, "*.mp4"))
        for f in files:
            feat = video_to_features(f)
            if feat is not None:
                X.append(feat)
                y.append(class_map[cls])
    
    X = np.array(X)
    y = tf.keras.utils.to_categorical(y, num_classes=len(classes))
    print(f"Loaded {X.shape[0]} videos with {len(classes)} classes")
    return X, y, classes

# ---------------- CNN + LSTM Model ----------------
def build_cnn_lstm(num_frames=NUM_FRAMES, feature_dim=1280, num_classes=100):
    inp = Input(shape=(num_frames, feature_dim))
    x = LSTM(256, return_sequences=False)(inp)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inp, out)
    return model

# ---------------- Main ----------------
if __name__ == "__main__":
    X, y, classes = load_dataset()
    if X.shape[0] == 0:
        raise RuntimeError("No valid videos found. Check your dataset!")
    
    model = build_cnn_lstm(num_frames=NUM_FRAMES, feature_dim=X.shape[2], num_classes=len(classes))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)
    
    # Save the trained model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "cnn_lstm_wlasl.h5")
    model.save(model_path)
    print(f"\n✅ Model saved at: {model_path}")
    plot_training_curves(history)

