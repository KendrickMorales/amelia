from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import wget
# import tarfile
from six.moves.urllib.request import urlretrieve
import zipfile
import os


def download_data():
    url = "https://docs.google.com/uc?export=download&id=10PXTNE82dZZkV7SQRiG38Pofe9QOeBBQ"
    filename, _ = urlretrieve(url, "data.zip")
    zfile = zipfile.ZipFile(filename)
    zfile.extractall(".")


def is_data_dowloaded():
    train_data = os.path.exists('data/train')
    test_data = os.path.exists('data/test')

    return (train_data and test_data)


def load_dataset(batch_size=16):

    if not is_data_dowloaded():
        download_data()

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(rescale=1. / 255.,
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


if __name__ == "__main__":
    load_dataset()
