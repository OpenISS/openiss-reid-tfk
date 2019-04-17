import random
import numpy as np
from collections import defaultdict

from .preprocess import load_image, img_to_array, imagenet_process

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
            if pid == 0: continue # pid = 0 means background
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
