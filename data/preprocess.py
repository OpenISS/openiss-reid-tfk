import random
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import to_categorical

to_categorical = to_categorical

def load_image(path, target_size):
    """
    Load the image into memory.

    Argument
        path: the path of the image can be relattvie or absolute
        target_size: the expected size of the return image in format (height, width)
    Return
        the loaded image
    """
    img = image.load_img(path, target_size=target_size)
    return img

def imagenet_process(ndarray):
    return preprocess_input(ndarray)

def img_to_array(img):
    return image.img_to_array(img)

def data_argumentation(img_list, pad, prob=0.5):
    """
    Apply data argumentation to a list of images. For each image:
        1. pad some pixels around the image
        2. randamly crop the image with the size before padding
        3. 0.5 probability to flip the image horizontally

    Argument
        pad: the number of pixels padding to the randonly selected images
        prob: the probability that will flip the image
    Return
        a list contains the argumented images
    """
    argumented_imgs = []
    for img in img_list:
        # randomly crop the image
        w_, h_ = img.size
        img = image.img_to_array(img)
        crop = image.random_shift(img, pad / w_, pad / h_,
            row_axis=0, col_axis=1, channel_axis=2,
            fill_mode='constant', cval=0)

        # flip the image horizontally
        flip_prob = random.random()
        if flip_prob > prob:
            crop = crop.swapaxes(1, 0)
            crop = crop[::-1, ...]
            crop = crop.swapaxes(0, 1)
        argumented_imgs.append(crop)

    return argumented_imgs
