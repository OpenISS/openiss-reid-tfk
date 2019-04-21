import random
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
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


def rea(imgs, img_w, img_h, ep=0.5, sl=0.02, sh=0.4,
        r1=0.3, r2=3.33, mean=(0.4914, 0.4822, 0.4465)):
    """
    Random Erasing Argumentation
    images: a mini-batch of images
    img_w:  the width of the batch images
    img_h:  the height of the batch iamges
    ep:     erasing probability
    sl, sh: erasing area ratio range sl and sh
    r1, r2: erasing aspect ratio range r1 and r2
    """
    p = np.random.rand()
    # print('p: {}'.format(p))
    if p > ep:
        return imgs
    else:
        s = img_w * img_h
        while True:
            se = np.random.uniform(sl, sh) * s
            re = np.random.uniform(r1, r2)
            he = int(np.sqrt(se * re))
            we = int(np.sqrt(se / re))
            xe = int(np.random.uniform(0.0, img_w))
            ye = int(np.random.uniform(0.0, img_h))
            if xe + we <= img_w and ye + he <= img_h:
                # pv = np.random.uniform(0.0, 255.0)
                for img in imgs:
                    img[0, xe:xe + we, ye:ye + he] = mean[0]
                    img[1, xe:xe + we, ye:ye + he] = mean[1]
                    img[2, xe:xe + we, ye:ye + he] = mean[2]
                return imgs

def sml(train_y_one_hot, num_classes, epsilon=0.1):
        is_one_mask = train_y_one_hot == 1
        is_zero_mask = train_y_one_hot == 0
        train_y_one_hot[is_one_mask] = 1 - ((num_classes - 1) / num_classes * epsilon)
        train_y_one_hot[is_zero_mask] = epsilon / num_classes
        return train_y_one_hot
