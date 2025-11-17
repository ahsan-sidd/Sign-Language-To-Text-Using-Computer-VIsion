import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
import os
import re

# ==================================
# 1) PATHS + CONFIG
# ==================================
TRAIN_DIR = "data/asl_alphabet_train/asl_alphabet_train"
TEST_DIR = "data/asl_alphabet_test/asl_alphabet_test"  # flat folder, no subfolders

IMG_SIZE = (128, 128)
BATCH_SIZE = 64
EPOCHS = 8

print("\nLoading dataset...\n")

# ==================================
# 2) TRAIN DATA
# ==================================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    label_mode="int"
)

class_names = train_ds.class_names
print("Classes:", class_names)

# PERFORMANCE BOOST
train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)

# ==================================
# 3) CUSTOM TEST LOADER (NO SUBFOLDERS)
# ==================================
def extract_label_from_filename(path):
    filename = tf.strings.split(path, os.sep)[-1]
    letter = tf.strings.regex_replace(filename, r"[^A-Za-z]", "")
    letter = tf.strings.upper(letter[0])    # take 1st letter
    idx = class_names.index(letter.numpy().decode())
    return idx

def load_test_images():
    filepaths = sorted([os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR)])
    images = []
    labels = []

    for p in filepaths:
        img_raw = tf.io.read_file(p)
        img = tf.image.decode_jpeg(img_raw, channels=3)
        img = tf.image.resize(img, IMG_SIZE) / 255.0
        images.append(img)

        # Extract label (A_test.jpg → A)
        letter = re.sub(r"[^A-Za-z]", "", os.path.basename(p))[0].upper()
        labels.append(class_names.index(letter))

    return tf.data.Dataset.from_tensor_slices((images, labels)).batch(BATCH_SIZE)

test_ds = load_test_images()

# ==================================
# 4) DATA AUGMENTATION
# ==================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ==================================
# 5) TRANSFER LEARNING MODEL
# ==================================
base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
base_model.trainable = False   # FREEZE first phase

inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ==================================
# 6) CALLBACKS (massive speed + accuracy boost)
# ==================================
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3)
]

# ==================================
# 7) STAGE 1 — TRAIN TOP LAYERS (FAST)
# ==================================
print("\n--- Training top layers ---\n")
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==================================
# 8) STAGE 2 — FINE-TUNING (unlock last 40 layers)
# ==================================
print("\n--- Fine tuning EfficientNet ---\n")
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),  # slower LR for fine-tuning
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_fine = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=5,
    callbacks=callbacks
)

model.save("asl_final_model.keras")
print("\nTraining Completed!\nModel saved as asl_final_model.keras")
