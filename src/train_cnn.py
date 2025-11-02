import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
from tensorflow.keras.optimizers import Adam
from data_loader import create_data_generators
from model_cnn import build_cnn_model
from utils.visualize_results import plot_training_curves

DATA_DIR = 'data/asl_alphabet_train/asl_alphabet_train'
MODEL_PATH = 'models/cnn_static.h5'
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10

def train_model():
    train_gen, val_gen = create_data_generators(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    model = build_cnn_model((*IMG_SIZE, 3), num_classes=train_gen.num_classes)

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, verbose=1)
    os.makedirs('models', exist_ok=True)
    model.save(MODEL_PATH)
    plot_training_curves(history)
    print(f"\n✅ Model saved at: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
