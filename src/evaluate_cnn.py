import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import create_data_generators

DATA_DIR = 'data/asl_alphabet_test/asl_alphabet_test'
MODEL_PATH = 'models/cnn_static.h5'
IMG_SIZE = (64, 64)
BATCH_SIZE = 32

def evaluate_model():
    _, val_gen = create_data_generators(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    model = load_model(MODEL_PATH)

    val_gen.reset()
    preds = model.predict(val_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=val_gen.class_indices.keys()))

    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    evaluate_model()
