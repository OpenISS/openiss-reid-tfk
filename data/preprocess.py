import random
import numpy as np
from keras.preprocessing import image
from backbone.resnet50 import preprocess_input
# from backbone.resnet50v2 import preprocess_input
from keras.utils import to_categorical

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
    return preprocess_input(np.asarray(ndarray))

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

    return np.asarray(argumented_imgs)


def rea(img, img_w, img_h, ep=0.5, sl=0.02, sh=0.4,
        r1=0.3, mean=[103.939, 116.779, 123.68]):
    """
    Random Erasing Argumentation
    img: an image in array format (h, w, ch)
    img_w:  the width of the batch images
    img_h:  the height of the batch iamges
    ep:     erasing probability
    sl, sh: erasing area ratio range sl and sh
    r1: erasing aspect ratio r1
    """
    p = np.random.rand()
    # print('p: {}'.format(p))
    if p > ep:
        return img
    else:
        s = img_w * img_h
        while True:
            se = np.random.uniform(sl, sh) * s
            re = np.random.uniform(r1, 1 / r1)
            he = int(round(np.sqrt(se * re)))
            we = int(round(np.sqrt(se / re)))
            xe = int(np.random.uniform(0.0, img_h))
            ye = int(np.random.uniform(0.0, img_w))
            if xe + he <= img_h and ye + we <= img_w:
                img[xe:xe + he, ye:ye + we, :] = mean
                return img
                # for img in imgs:
                    # img[xe:xe + we, ye:ye + he, :] = mean

                # pv = np.random.uniform(0.0, 255.0)
                # for img in imgs:
                #     img[xe:xe + we, ye:ye + he, :] = pv
                # return img

def sml(train_y_one_hot, num_classes, epsilon=0.1):
    is_one_mask = train_y_one_hot == 1
    is_zero_mask = train_y_one_hot == 0
    train_y_one_hot[is_one_mask] = 1 - ((num_classes - 1) / num_classes * epsilon)
    train_y_one_hot[is_zero_mask] = epsilon / num_classes
    return train_y_one_hot
