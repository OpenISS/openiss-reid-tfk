# -* - coding: UTF-8 -* -

import random
import os

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
        self.ids, self.id2paths = self._process(dataset)
        random.shuffle(self.ids)
        self.len = len(self.ids)
        self.backup_ids = []
        self.refill_threshold = self.num_ids_per_batch - 1


    def batch_data(self):
        return self._rand_select_images()

    def _rand_select_images(self):
        """
        Randomly select p identities from the dataset, for each identity
        randomly pick k images.

        Return
            a list of images and a list of corresponding pids of these images
        """
        image_list = []
        label_list = []
        # loop over the pid in self.ids, randomly select the negative samples
        cur_id = self.ids.pop()
        sample_ids = [cur_id]
        neg_ids = random.sample(self.ids, self.num_ids_per_batch - 1)
        sample_ids.extend(neg_ids)
        assert(len(sample_ids) == self.num_ids_per_batch)

        for iid in sample_ids:
            available_paths = self.id2paths[iid]
            selected_path = random.sample(
                available_paths, self.num_imgs_per_id)

            # for each random id, select k image randomly
            for path in selected_path:
                img = load_image(path, target_size=(self.img_h, self.img_w))
                image_list.append(img)
                label_list.append(int(iid))

        # reset the index for the next epoch
        self.backup_ids.append(cur_id)
        if len(self.ids) <= self.refill_threshold:
            self.backup_ids.extend(self.ids)    # get all remaining pids
            self.ids = self.backup_ids          # update ids reference
            random.shuffle(self.ids)            # shuffle the pids list
            self.backup_ids = []                # create new memory for backup

        return image_list, label_list

    def _process(self, dataset):
        """
        Prepare data structure for the sampling and eliminate the useless data.

        Argument
            dataset: the dataset used for current training or validation
        """
        # create a dictionary with the following structure
        #  {  pid1 : [path11, path12, ...],
        #     pid2 : [path21, path22, ...], ... }
        id2paths = {}
        for data in dataset:
            image_path, pid, _ = data
            if pid in id2paths:
                id2paths[pid].append(image_path)
            else:
                id2paths[pid] = [image_path]

        # eliminate the pid which contains less than the required k images
        ids = []
        for pid, path_list in id2paths.items():
            if len(path_list) >= self.num_imgs_per_id:
                ids.append(pid)
        print('[sampler] pids in the dataset: {}'.format(len(id2paths.keys())))
        print('[sampler] useable pids: {}'.format(len(ids)))
        return ids, id2paths
