from keras.preprocessing.image import ImageDataGenerator


def download_data():
    pass


def is_data_dowloaded():
    pass


def load_dataset(batch_size=16):

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            'data/train',  # this is the target directory
            target_size=(300, 300),  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    test_generator = test_datagen.flow_from_directory(
            'data/test',
            target_size=(300, 300),
            batch_size=batch_size,
            class_mode='binary')

    return train_generator, test_generator
