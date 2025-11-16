import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, Rescaling
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from asl_data_prep import get_data_generators, TARGET_SIZE, NUM_CLASSES, BATCH_SIZE

# --- Model Building Functions ---

def build_feature_extraction_model(input_shape, num_classes):
    """
    Builds a Transfer Learning model using MobileNetV2 as a frozen feature extractor.
    
    The pre-trained weights are from ImageNet.
    """
    
    # 1. Define the input layer
    inputs = Input(shape=input_shape)
    
    # 2. Add MobileNetV2 preprocessing: Rescales [0, 1] input to [-1, 1]
    # This is a key step to match the range the pre-trained model was trained on.
    x = Rescaling(1./127.5, offset=-1)(inputs) 
    
    # 3. Load the MobileNetV2 base model
    # weights='imagenet': Loads weights pre-trained on the ImageNet dataset.
    # include_top=False: Excludes the original classification layer.
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # 4. Freeze the base model layers
    # This is the essence of *Feature Extraction* transfer learning
    base_model.trainable = False
    
    # 5. Pass the input through the frozen base model
    # training=False ensures that batch normalization layers remain frozen as well
    x = base_model(x, training=False)
    
    # 6. Add a new classification head (our trainable layers)
    x = GlobalAveragePooling2D()(x) # Condenses feature maps into a vector
    x = Dropout(0.5)(x)             # Regularization to prevent overfitting
    
    # Final classification layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # 7. Create the final model
    model = Model(inputs, outputs)
    
    return model

def build_fine_tuning_model(base_model, fine_tune_from_layer=100):
    """
    Takes a model (e.g., the feature extraction model) and unfreezes
    the top layers for fine-tuning.
    
    Note: The MobileNetV2 base has 154 total layers.
    """
    # Unfreeze the base model
    base_model.trainable = True
    
    # Freeze all layers up to a certain point (e.g., layer 100)
    for layer in base_model.layers[:fine_tune_from_layer]:
        # Skip the input and rescaling layer
        if not isinstance(layer, (Input, Rescaling)): 
            layer.trainable = False

    # The remaining top layers of the base model will be trainable
    
    return base_model


# --- Training Workflow ---

def train_model():
    """Main function to run the transfer learning and fine-tuning workflow."""
    
    # --- Data Preparation ---
    train_generator, validation_generator = get_data_generators()
    
    # --- 1. Feature Extraction (Transfer Learning) ---
    print("\n--- Starting Feature Extraction (Transfer Learning) ---")
    
    feature_model = build_feature_extraction_model(
        input_shape=TARGET_SIZE + (3,), # (224, 224, 3)
        num_classes=NUM_CLASSES
    )

    feature_model.compile(
        optimizer=Adam(learning_rate=0.0001), # Small learning rate is safer for transfer learning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print the model summary to see which layers are frozen/trainable
    print("\nFeature Extraction Model Summary:")
    feature_model.summary(expand_nested=True)
    
    history_feature = feature_model.fit(
        train_generator,
        epochs=10, 
        validation_data=validation_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_steps=validation_generator.samples // BATCH_SIZE
    )

    # --- 2. Fine-Tuning (Optional but recommended) ---
    print("\n--- Starting Fine-Tuning ---")
    
    # Re-compile the model with a much smaller learning rate
    # A smaller LR is crucial to avoid corrupting the pre-trained weights
    
    fine_tune_model = build_fine_tuning_model(feature_model)
    
    # Print summary to confirm which layers are now unfrozen (trainable)
    print("\nFine-Tuning Model Summary (Check Trainable Params):")
    fine_tune_model.summary(expand_nested=True)
    
    fine_tune_model.compile(
        optimizer=Adam(learning_rate=1e-5), # Very small learning rate for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training for a few more epochs
    history_fine_tune = fine_tune_model.fit(
        train_generator,
        epochs=10, # Additional 10 epochs
        initial_epoch=history_feature.epoch[-1], # Start from the last epoch of previous training
        validation_data=validation_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_steps=validation_generator.samples // BATCH_SIZE
    )
    
    # --- Evaluation ---
    print("\n--- Final Evaluation ---")
    loss, accuracy = fine_tune_model.evaluate(validation_generator)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Save the final model
    fine_tune_model.save('asl_mobilenetv2_transfer_learning.h5')
    print("Model saved as 'asl_mobilenetv2_transfer_learning.h5'")


if __name__ == '__main__':
    # Ensure you have the ASL Alphabet Dataset downloaded and the path is set 
    # correctly in asl_data_prep.py before running.
    train_model()