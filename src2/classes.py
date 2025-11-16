import os
import json
import shutil

# Paths
JSON_PATH = "./data/WLASL/start_kit/WLASL_v0.3.json"
VIDEOS_DIR = "./data/WLASL/start_kit/videos"           # your extracted clips folder
OUTPUT_DIR = "./data/WLASL/start_kit/dataset"          # final dataset folder

def organize_wlasl():
    # Load JSON metadata
    with open(JSON_PATH, 'r') as f:
        content = json.load(f)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    count = 0

    for entry in content:
        gloss = entry["gloss"]             # THIS IS THE CLASS LABEL (e.g. "BOOK", "HELLO")
        instances = entry["instances"]

        # Create class folder
        class_folder = os.path.join(OUTPUT_DIR, gloss)
        os.makedirs(class_folder, exist_ok=True)

        for inst in instances:
            video_id = inst["video_id"]    # each instance has a unique video file name
            filename = f"{video_id}.mp4"

            src_path = os.path.join(VIDEOS_DIR, filename)
            dst_path = os.path.join(class_folder, filename)

            # Only copy files that exist (some videos missing)
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
                count += 1
                print(f"Copied {filename} → {gloss}")
            else:
                print(f"Missing video file: {filename}")

    print(f"\n✅ Done! Total videos organized: {count}")

if __name__ == "__main__":
    organize_wlasl()
