import os
import glob
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2

# ---------------- Configuration ----------------
DATASET_DIR = "data/WLASL/start_kit/dataset"
MIN_VIDEOS_PER_CLASS = 5
NUM_FRAMES = 16
IMG_SIZE = (224, 224)
BATCH_SIZE = 4
EPOCHS = 10

# ---------------- Utilities ----------------
def sample_frames(video_path, num_frames=NUM_FRAMES):
    """
    Safely sample frames from video.
    Skip corrupted frames.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            return None

        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = []
        current = 0

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.resize(frame, IMG_SIZE)
            frames.append(frame / 255.0)

        cap.release()

        if len(frames) == 0:
            return None

        while len(frames) < num_frames:
            frames.append(frames[-1])

        return np.array(frames)

    except Exception:
        return None


# ---------------- Feature Extractor ----------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(*IMG_SIZE, 3)
)
feature_extractor = Model(base_model.input, base_model.output)


def video_to_features(path):
    frames = sample_frames(path)
    if frames is None:
        return None
    return feature_extractor.predict(frames, verbose=0)


# ---------------- Load dataset ----------------
def load_dataset():
    class_counts = {}
    for cls in os.listdir(DATASET_DIR):
        n = len(glob.glob(os.path.join(DATASET_DIR, cls, "*.mp4")))
        class_counts[cls] = n

    valid_classes = [c for c, n in class_counts.items() if n >= MIN_VIDEOS_PER_CLASS]
    print(f"✔ Using {len(valid_classes)} classes out of 2000 (filtered by ≥{MIN_VIDEOS_PER_CLASS} videos)")

    X, y = [], []
    class_map = {cls: i for i, cls in enumerate(sorted(valid_classes))}

    for cls in valid_classes:
        video_files = glob.glob(os.path.join(DATASET_DIR, cls, "*.mp4"))
        for vid in video_files:
            feat = video_to_features(vid)
            if feat is not None:
                X.append(feat)
                y.append(class_map[cls])

    X = np.array(X)
    y = tf.keras.utils.to_categorical(y, num_classes=len(valid_classes))

    print(f"Loaded {X.shape[0]} videos with {len(valid_classes)} classes")
    return X, y, valid_classes


# ---------------- Model ----------------
def build_model(num_frames, feature_dim, num_classes):
    inp = Input(shape=(num_frames, feature_dim))
    x = LSTM(256)(inp)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation="softmax")(x)
    model = Model(inp, out)
    return model


# ---------------- Main ----------------
if __name__ == "__main__":
    X, y, classes = load_dataset()

    if X.shape[0] == 0:
        raise RuntimeError("No valid videos found after filtering.")

    model = build_model(
        num_frames=NUM_FRAMES,
        feature_dim=X.shape[2],
        num_classes=len(classes)
    )

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(
        X, y,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    # Save the model in modern format
    model.save("models/cnn_lstm_subset.keras")
    print("✔ Model saved as cnn_lstm_subset.keras")

    # Plot accuracy & loss
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()

    plt.savefig("training_curves.png")
    print("✔ Saved training plot as training_curves.png")
    plt.show()
