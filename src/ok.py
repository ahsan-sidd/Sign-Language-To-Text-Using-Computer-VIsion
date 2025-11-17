import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import numpy as np
import cv2

# ===============================
# 1️⃣ Dataset Loading
# ===============================
TRAIN_DIR = "data/asl_alphabet_train/asl_alphabet_train"
TEST_DIR = "data/asl_alphabet_test/asl_alphabet_test"  # flat folder, no subfolders

IMG_SIZE = (200, 200)
BATCH_SIZE = 32

# Train dataset (folders per class)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    shuffle=True
)

# Validation dataset (from train with 20% split)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    validation_split=0.2,
    subset='validation',
    seed=42
)

# Test dataset (flat folder, parse class from filename)
test_files = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Extract labels from filenames (e.g., "A_test123.jpg" -> "A")
test_labels = [f.split(os.sep)[-1].split('_')[0].upper() for f in test_files]

class_names = train_ds.class_names
class_to_index = {name: i for i, name in enumerate(class_names)}
test_indices = [class_to_index.get(lbl, 0) for lbl in test_labels]  # default to 0 if not found

import cv2
import tensorflow as tf
import numpy as np
import os

def decode_img_cv(path):
    # Read image using OpenCV
    img = cv2.imread(path.numpy().decode())  # path comes as a tensor, convert to string
    if img is None:
        # return a blank image if read fails
        img = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
    else:
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize
        img = cv2.resize(img, IMG_SIZE)
    # Normalize to [0,1]
    img = img / 255.0
    return img.astype(np.float32)

def load_sample(path, label):
    img = tf.py_function(decode_img_cv, [path], tf.float32)
    img.set_shape((*IMG_SIZE, 3))  # set known shape
    return img, label

# Example usage:
# test_ds = tf.data.Dataset.from_tensor_slices((test_file_paths, test_labels))
# test_ds = test_ds.map(load_sample_cv).batch(BATCH_SIZE)


test_ds = tf.data.Dataset.from_tensor_slices((test_files, test_indices))
test_ds = test_ds.map(load_sample).batch(BATCH_SIZE)

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# ===============================
# 2️⃣ Data Augmentation
# ===============================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ===============================
# 3️⃣ Transfer Learning Model
# ===============================
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3))
base_model.trainable = False  # freeze base

inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===============================
# 4️⃣ Callbacks
# ===============================
checkpoint = ModelCheckpoint("asl_best_model.h5", save_best_only=True, monitor='val_accuracy', mode='max')
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# ===============================
# Load checkpoint if exists
# ===============================
if os.path.exists("asl_best_model.h5"):
    print("\nFound checkpoint → loading previous weights...\n")
    model = tf.keras.models.load_model("asl_best_model.h5")
else:
    print("\nNo checkpoint found → training from scratch...\n")


# ===============================
# 5️⃣ Train the Model
# ===============================
EPOCHS = 0
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

# ===============================
# 6️⃣ Fine-tuning
# ===============================
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=[checkpoint, early_stop]
)

# ===============================
# 7️⃣ Evaluate on Test Set
# ===============================
test_loss, test_acc = model.evaluate(test_ds)
print("Test Accuracy:", test_acc)

# ===============================
# 8️⃣ Live Webcam Inference
# ===============================
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, IMG_SIZE)
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    preds = model.predict(img_array, verbose=0)
    class_idx = np.argmax(preds)
    class_name = class_names[class_idx]

    cv2.putText(frame, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
