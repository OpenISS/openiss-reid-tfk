# -* - coding: UTF-8 -* -

import random
import os
import copy
import numpy as np
from collections import defaultdict
from .preprocess import load_image

class RandomSampler(object):

    def __init__(self, dataset, num_ids, num_imgs_per_id, img_w, img_h):
        """
        Arguments
            dataset: the dataset object which contains the train, gallery and query
                     each of them will be a list of (image_path, pid, camid).
            cfg: the configuration object.
        """
        self.num_ids_per_batch = num_ids
        self.num_imgs_per_id = num_imgs_per_id
        self.img_w = img_w
        self.img_h = img_h
        self.counter = 0

        # map the id to all its paths
        self.id2paths = defaultdict(list)
        for (path, pid, _) in dataset:
            self.id2paths[pid].append(path)
        self.ids = list(self.id2paths.keys())

        # how many images we have in an epoch
        self.len = 0
        for pid in self.ids:
            id_paths = self.id2paths[pid]
            s_num = len(id_paths)
            if s_num < self.num_imgs_per_id:
                s_num = self.num_imgs_per_id
            self.len += s_num - (s_num % self.num_imgs_per_id)


    def batch_data(self):
        self.counter += 1
        return self._rand_select_images()


    def _rand_select_images(self):
        """
        Randomly select p identities from the dataset, for each identity
        randomly pick k images.

        Return
            a list of images and a list of corresponding pids of these images
        """
        img_paths = []
        img_labels = []
        # loop over the pid in self.ids, randomly select the negative samples
        cur_id = self.ids.pop()
        sample_ids = [cur_id]
        neg_ids = random.sample(self.ids, self.num_ids_per_batch - 1)
        sample_ids.extend(neg_ids)
        assert(len(sample_ids) == self.num_ids_per_batch)

        for iid in sample_ids:
            avai_paths = self.id2paths[iid]
            # if avai_paths for current pid is less than the needed value
            if len(avai_paths) < self.num_imgs_per_id:
                avai_paths = list(np.random.choice(avai_paths, size=self.num_imgs_per_id, replace=True))

            # for each random id, select k image randomly
            selected_path = random.sample(avai_paths, self.num_imgs_per_id)

            for path in selected_path:
                img_paths.append(path)
                img_labels.append(int(iid))

        self.ids.append(cur_id)
        return img_paths, img_labels