import cv2
import numpy as np
from tensorflow.keras.models import load_model
from data_loader import create_data_generators

MODEL_PATH = 'models/cnn_static.h5'
IMG_SIZE = (64, 64)

# Use generator once to get class labels
_, val_gen = create_data_generators('data/asl_alphabet_train/asl_alphabet_train', IMG_SIZE)
labels = list(val_gen.class_indices.keys())

def run_realtime():
    model = load_model(MODEL_PATH)
    cap = cv2.VideoCapture(0)

    print("🎥 Press 'q' to quit the live demo.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = cv2.resize(frame, IMG_SIZE)
        roi = roi / 255.0
        roi = np.expand_dims(roi, axis=0)
        pred = model.predict(roi)
        label = labels[np.argmax(pred)]

        cv2.putText(frame, f"Prediction: {label}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Sign Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime()
