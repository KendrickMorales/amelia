from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import wget
# import tarfile
from six.moves.urllib.request import urlretrieve
import zipfile
import os
import glob
import tensorflow as tf

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

#     if not is_data_dowloaded():
#         download_data()

    classes = os.listdir('../images/')
    sorted(classes)

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(rescale=1. / 255.,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       brightness_range=[-0.2, 0.2],
                                       validation_split=0.2,
                                       horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            '../images/',  # this is the target directory
            target_size=(384, 384),  # all images will be resized to 150x150
            batch_size=batch_size,
            classes=classes,
#             classes=["negative", "positive"],
            class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
#     test_generator = test_datagen.flow_from_directory(
#             '../images',
#             target_size=(300, 300),
#             batch_size=batch_size,
#             classes=["negative", "positive"],
#             class_mode='binary')

    return train_generator


# def load_dataset_from_folder(folder):
#     classes = os.listdir(folder)
#     sorted(classes)
#     paths = list()
#     y = list()
    
#     for i, food in enumerate(classes):
#         cpath = os.path.join(folder, food)
#         food_images = os.listdir(cpath)
        
#         for image in food_images:
#             image_path = os.path.join(cpath, image)
#             paths.append(image_path)
#             y.append(i)
    
#     X = np.array(paths)
#     y = np.array(y)
    
def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, '/')
    # The second to last is the class-directory
    return parts[-2] == CLASS_NAMES


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)/255.
    return img, label


def load_dataset_from_folder(folder, parallel_num=4):
    filenames = glob.glob("../images/*/*")
    
    list_ds = tf.data.Dataset.list_files(filenames)
    
    labeled_ds = list_ds.map(process_path, num_parallel_calls=parallel_num)
    
    return labeled_ds
    

if __name__ == "__main__":
    load_dataset()
