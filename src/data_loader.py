from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generators(data_dir, img_size=(64, 64), batch_size=32, val_split=0.2):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,   # <--- IMPORTANT FOR ASL
        fill_mode='nearest',
        validation_split=0.2
    )



    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_gen, val_gen
