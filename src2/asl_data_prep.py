import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- Configuration ---
DATASET_PATH = 'data/asl_alphabet_test/asl_alphabet_test' # <-- CHANGE THIS
TARGET_SIZE = (224, 224) # MobileNetV2 standard input size
BATCH_SIZE = 32
NUM_CLASSES = 29 # A-Z, plus 'del', 'nothing', 'space' in the common Kaggle dataset
VALIDATION_SPLIT = 0.2

def get_data_generators(data_dir=DATASET_PATH, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT):
    """
    Creates training and validation data generators for the ASL dataset.
    
    The preprocessing function from MobileNetV2 is crucial here: it scales
    pixel values to the range [-1, 1] as expected by the pre-trained model.
    """
    # Create an ImageDataGenerator with the MobileNetV2 preprocessing function
    # Note: We only apply a rescale on the base ImageDataGenerator, and rely on the 
    # MobileNetV2.preprocess_input when using flow_from_directory with a function
    # or by applying it explicitly in the model.
    # A cleaner approach is to use the Keras utility layers, but for flow_from_directory, 
    # we'll use the in-built MobileNetV2 preprocess_input to ensure correct scaling.
    
    # We will use the built-in ImageDataGenerator preprocessing for simplicity 
    # and then rely on a Rescaling layer in the model for the MobileNetV2-specific scaling
    # The initial rescale=1./255. is a common base step.
    
    datagen = ImageDataGenerator(
        rescale=1./255., # Initial pixel scaling to [0, 1]
        validation_split=validation_split,
        # Common data augmentation techniques
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        fill_mode='nearest'
    )

    # Training Generator
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        seed=42
    )
    
    # Validation Generator
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        seed=42
    )

    return train_generator, validation_generator

if __name__ == '__main__':
    # Test the generators
    train_gen, val_gen = get_data_generators()
    print(f"Total training images: {train_gen.samples}")
    print(f"Total validation images: {val_gen.samples}")
    print(f"Class indices: {train_gen.class_indices}")

    # You can save the class indices for later mapping predictions back to letters
    import json
    with open('class_indices.json', 'w') as f:
        json.dump(train_gen.class_indices, f)