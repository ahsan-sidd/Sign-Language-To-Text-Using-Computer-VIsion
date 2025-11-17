"""
Pose / Keypoints + LSTM for WLASL
"""

import os
import glob
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

DATASET_DIR = "data/WLASL/start_kit/dataset"
NUM_FRAMES = 32

# --- Utils ---
mp_holistic = mp.solutions.holistic

def extract_keypoints(video_path, num_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    seq = []
    holistic = mp_holistic.Holistic(static_image_mode=False)
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        if i in indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            keypoints = []
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
            seq.append(keypoints)
    cap.release()
    return np.array(seq)

def load_dataset():
    X, y = [], []
    classes = sorted(os.listdir(DATASET_DIR))
    class_map = {cls:i for i, cls in enumerate(classes)}
    for cls in classes:
        files = glob.glob(os.path.join(DATASET_DIR, cls, "*.mp4"))
        for f in files:
            keypoints_seq = extract_keypoints(f)
            if keypoints_seq.shape[0] == NUM_FRAMES:
                X.append(keypoints_seq)
                y.append(class_map[cls])
    X = np.array(X)
    y = tf.keras.utils.to_categorical(y, num_classes=len(classes))
    return X, y, classes

# --- Model ---
def build_pose_lstm_model(num_frames=NUM_FRAMES, num_features=258, num_classes=100):
    inp = Input(shape=(num_frames, num_features))
    x = LSTM(256, return_sequences=True)(inp)
    x = LSTM(128)(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation='softmax')(x)
    return Model(inp, out)

# --- Main ---
if __name__ == "__main__":
    X, y, classes = load_dataset()
    print(f"Loaded {X.shape[0]} videos, {len(classes)} classes")
    
    num_features = X.shape[2]
    model = build_pose_lstm_model(num_features=num_features, num_classes=len(classes))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, batch_size=4, epochs=10, validation_split=0.1)
