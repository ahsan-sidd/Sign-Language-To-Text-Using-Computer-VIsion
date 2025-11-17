import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from stgcn_model import build_stgcn

DATASET = "keypoints"
TARGET_FRAMES = 75  # adjust to what your ST-GCN expects
BATCH_SIZE = 16
EPOCHS = 30

def pad_sequence(seq, target_len=TARGET_FRAMES):
    """
    Pads or truncates a keypoint sequence to target_len frames.
    seq: (frames, joints, coords)
    Returns: (target_len, joints, coords)
    """
    frames, joints, coords = seq.shape
    if frames >= target_len:
        return seq[:target_len]
    padding = np.zeros((target_len - frames, joints, coords))
    return np.vstack([seq, padding])

def load_dataset():
    X, y = [], []
    classes = sorted(os.listdir(DATASET))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    total_files = 0
    loaded_files = 0

    for cls in classes:
        folder = os.path.join(DATASET, cls)
        if not os.path.isdir(folder):
            continue

        files = sorted(os.listdir(folder))
        print(f"\nClass {cls} | Files: {len(files)}")
        for f in files:
            total_files += 1
            path = os.path.join(folder, f)
            arr = np.load(path)
            if arr.ndim != 3:
                print(f"❌ Skipped {f} — invalid dims {arr.shape}")
                continue

            arr = pad_sequence(arr, TARGET_FRAMES)
            X.append(arr)
            y.append(class_to_idx[cls])
            loaded_files += 1

    X = np.array(X)  # shape: (N, T, V, C)
    y = np.array(y)
    print(f"\nTotal files: {total_files}, Loaded samples: {loaded_files}")
    print(f"Shapes: X={X.shape}, y={y.shape}")
    return X, y, classes

if __name__ == "__main__":
    X, y, classes = load_dataset()

    num_classes = len(classes)
    print("Loaded:", X.shape, "Classes:", num_classes)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    model = build_stgcn(T=X.shape[1], V=X.shape[2], C=X.shape[3],
                        num_classes=num_classes)
    model.summary()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint = ModelCheckpoint(
        "models/stgcn_best.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    early = EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True)

    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint, early]
    )

    # ---- Plotting ----
    plt.figure(figsize=(10,5))
    plt.plot(hist.history["accuracy"], label="train acc")
    plt.plot(hist.history["val_accuracy"], label="val acc")
    plt.legend()
    plt.title("ST-GCN Accuracy")
    plt.savefig("stgcn_accuracy.png")

    plt.figure(figsize=(10,5))
    plt.plot(hist.history["loss"], label="train loss")
    plt.plot(hist.history["val_loss"], label="val loss")
    plt.legend()
    plt.title("ST-GCN Loss")
    plt.savefig("stgcn_loss.png")

    print("Training complete. Model saved to models/stgcn_best.keras")
