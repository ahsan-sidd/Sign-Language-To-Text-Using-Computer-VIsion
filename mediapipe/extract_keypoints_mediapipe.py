import cv2
import os
import numpy as np
import mediapipe as mp
from tqdm import tqdm

DATASET_DIR = "./data/WLASL/start_kit/dataset"
SAVE_DIR = "keypoints"
os.makedirs(SAVE_DIR, exist_ok=True)

mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    pose = np.zeros(33 * 4)
    left_hand = np.zeros(21 * 3)
    right_hand = np.zeros(21 * 3)

    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            pose[i*4:(i+1)*4] = [lm.x, lm.y, lm.z, lm.visibility]

    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            left_hand[i*3:(i+1)*3] = [lm.x, lm.y, lm.z]

    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            right_hand[i*3:(i+1)*3] = [lm.x, lm.y, lm.z]

    return np.concatenate([pose, left_hand, right_hand])


def process_video(video_path, output_path):
    # RESUME SUPPORT: If file exists → skip
    if os.path.exists(output_path):
        return

    cap = cv2.VideoCapture(video_path)
    sequence = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            sequence.append(extract_keypoints(results))

    cap.release()
    
    # Save keypoints
    np.save(output_path, np.array(sequence))


# ---- MAIN LOOP ----

classes = sorted(os.listdir(DATASET_DIR))

print(f"Found {len(classes)} classes")

for cls in classes:
    class_input_dir = os.path.join(DATASET_DIR, cls)
    class_output_dir = os.path.join(SAVE_DIR, cls)

    os.makedirs(class_output_dir, exist_ok=True)

    videos = [v for v in os.listdir(class_input_dir) if v.endswith(".mp4")]

    print(f"\nProcessing class: {cls} ({len(videos)} videos)")

    for video in tqdm(videos):
        input_path = os.path.join(class_input_dir, video)
        output_path = os.path.join(class_output_dir, video.replace(".mp4", ".npy"))

        process_video(input_path, output_path)

print("DONE — All keypoints extracted.")
