# -* - coding: UTF-8 -* -

import random
import os
import numpy as np

from .sampler import RandomSampler
from .preprocess import data_argumentation, imagenet_process, load_image, img_to_array

class DataGen(object):

    def __init__(self, dataset, num_ids, num_imgs_per_id, img_w, img_h):
        """
        Constructor of DataGen object (this object will be used for keras as the
        blueprint of the data generator).

        Arguments
            dataset: train data in the format of (image_path, pid, camid).
        """
        self.padding = 10
        self.sampler = RandomSampler(dataset,
            num_ids, num_imgs_per_id,
            img_w, img_h
            )

    def flow(self):
        """
        Interface for the keras data generator.

        Argument
            padding: the number of pixels will be used for data argumentation
        Return
            a batch of data with or without data argumentation
        """
        batch_imgs, batch_pids = self.sampler.batch_data()
        batch_imgs = data_argumentation(batch_imgs, self.padding)

        batch_imgs = imagenet_process(np.asarray(batch_imgs))
        batch_pids = np.asarray(batch_pids)
        return batch_imgs, batch_pids