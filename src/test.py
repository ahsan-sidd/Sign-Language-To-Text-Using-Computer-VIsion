import cv2
import os
import re
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ===============================
# CONFIG
# ===============================
TEST_DIR = "data/asl_alphabet_test/asl_alphabet_test"  # flat folder
IMG_SIZE = (200, 200)
MODEL_PATH = "asl_best_model.h5"

CLASS_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G',
    'H', 'I', 'J', 'K', 'L', 'M', 'N',
    'O', 'P', 'Q', 'R', 'S', 'T', 'U',
    'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

# ===============================
# LOAD MODEL
# ===============================
model = tf.keras.models.load_model(MODEL_PATH)

# ===============================
# HELPER FUNCTIONS
# ===============================
def preprocess(img):
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def get_label_from_filename(filename):
    letter = re.sub(r"[^A-Za-z]", "", filename)
    letter = letter[0].upper()  # first letter as label
    return CLASS_NAMES.index(letter)

# ===============================
# TESTING
# ===============================
y_true = []
y_pred = []

for filename in sorted(os.listdir(TEST_DIR)):
    filepath = os.path.join(TEST_DIR, filename)
    img = cv2.imread(filepath)
    if img is None:
        continue

    label_true = get_label_from_filename(filename)
    img_input = preprocess(img)
    pred_probs = model.predict(img_input, verbose=0)
    label_pred = np.argmax(pred_probs)

    y_true.append(label_true)
    y_pred.append(label_pred)

    # Display image with prediction
    cv2.putText(img, f"Pred: {CLASS_NAMES[label_pred]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    cv2.imshow("ASL Test", img)
    key = cv2.waitKey(100) & 0xFF  # 100ms per image, press 'q' to quit
    if key == ord('q'):
        break

cv2.destroyAllWindows()

# ===============================
# METRICS
# ===============================
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
cm = confusion_matrix(y_true, y_pred)

print(f"\nAccuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1 Score: {f1*100:.2f}%")
print("\nConfusion Matrix:")
print(cm)
