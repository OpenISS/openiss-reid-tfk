# -* - coding: UTF-8 -* -
import random
import os
import numpy as np

from collections import defaultdict
from .sampler import RandomSampler
from .preprocess import data_argumentation, imagenet_process, \
                        load_image, img_to_array, rea, sml

class DataGen(object):

    def __init__(self, dataset, num_ids, num_imgs_per_id, img_w, img_h):
        """
        Constructor of DataGen object (this object will be used for keras as the
        blueprint of the data generator).

        Arguments
            dataset: train data in the format of (image_path, pid, camid).
        """
        self.padding = 10
        self.dataset = dataset
        self.sampler = RandomSampler(dataset,
            num_ids, num_imgs_per_id,
            img_w, img_h
            )
        self.img_h = img_h
        self.img_w = img_w

    def flow(self):
        """
        Interface for the keras data generator.

        Argument
            padding: the number of pixels will be used for data argumentation
        Return
            a batch of data with or without data argumentation
        """
        batch_imgs_path, batch_pids = self.sampler.batch_data()

        # form an image batch
        batch_imgs = []
        for path in batch_imgs_path:
            img = load_image(path, target_size=(self.img_h, self.img_w))
            batch_imgs.append(img)

        # enable data argumentation
        batch_imgs = data_argumentation(batch_imgs, self.padding)
        # enable random erasing
        batch_imgs = rea(batch_imgs, self.img_w, self.img_h)
        # adapt to imagenet
        batch_imgs = imagenet_process(batch_imgs)
        # enable smooth label
        # batch_pids = sml(batch_pids)
        batch_pids = np.asarray(batch_pids)
        return batch_imgs, batch_pids


class ValDataGen:
    def __init__(self, q_ds, g_ds, target_size):
        self.num_ids = 16
        self.num_imgs = 4
        self.target_size = target_size

        self.q_ds = q_ds
        self.g_ds = g_ds
        self.ids, self.q_id2tuple, self.g_id2tuple \
            = self._gather_info(self.q_ds, self.g_ds, self.num_imgs / 2)

    def flow(self):
        num_per_ds = int(self.num_imgs / 2)

        # randomly select p ids
        selected_pids = random.sample(self.ids, self.num_ids)
        # select query and gallery images for the selected ids
        # TODO: never make sure for the same pid in query and gallery
        #       with a different camid
        all_selected = []

        for pid in selected_pids:
            q_tuple_list = self.q_id2tuple[pid]
            g_tuple_list = self.g_id2tuple[pid]
            # always add query first, we need to know which
            # are query and which are gallery to calc acc
            q = random.sample(q_tuple_list, num_per_ds)
            g = random.sample(g_tuple_list, num_per_ds)
            all_selected += q + g

        # make image batch and label batch
        # assert len(selected_pids) == len(all_selected), \
        #     'length of two lists mismatched: {} and {}'.format(
        #         len(selected_pids), len(all_selected))
        images = []
        labels = []
        cams = []
        for item in all_selected:
            path, pid, camid = item
            img = load_image(path, self.target_size)
            img = img_to_array(img)
            images.append(img)
            labels.append(pid)
            cams.append(camid)

        images = imagenet_process(images)
        images = np.asarray(images)
        labels = np.asarray(labels)
        cams = np.asarray(cams)
        return images, labels, cams

    def _gather_info(self, q_ds, g_ds, min_len):
        q_id2tuple = defaultdict(list)
        for ds_tuple in q_ds:
            path, pid, camid = ds_tuple
            q_id2tuple[pid].append((path, pid, camid))
        g_id2tuple = defaultdict(list)
        for ds_tuple in g_ds:
            path, pid, camid = ds_tuple
            if pid == 0:
                continue  # pid = 0 means background
            g_id2tuple[pid].append((path, pid, camid))
        assert len(q_id2tuple.keys()) == len(g_id2tuple.keys()), \
            'query and gallery pids mismatched, got {} in query but {} in gallery'.format(
                len(q_id2tuple.keys()), len(g_id2tuple.keys()))
        ids = []
        for key in q_id2tuple.keys():
            q_val_len = len(q_id2tuple[key])
            g_val_len = len(g_id2tuple[key])
            if q_val_len > min_len and g_val_len > min_len:
                ids.append(key)
        return ids, q_id2tuple, g_id2tuple
