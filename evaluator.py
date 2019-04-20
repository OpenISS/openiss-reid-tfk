import os
import math
import h5py
import numpy as np

import keras.backend as K
from keras.utils import normalize

from data.preprocess import load_image, imagenet_process, img_to_array


def load_img_to_array(path, target_size):
    img = load_image(path, target_size)
    img = img_to_array(img)
    return img

class Evaluator:
    """
    Evaluator is used to compute the CMC and mAP metrics for person re-identification problem.

    Procedure:
        1. using the model to calculate all the images in gallery to obtain their feature maps
        2. using the model to calculate feature maps of all query images
        3. for each query image feature map, calculate the distance against all the gallery
           images feature maps, forming the distance matrix which shape is [num_query, num_gallery]
    """

    def __init__(self, dataset, model, img_h, img_w):
        self.dataset = dataset
        self.model = model
        self.img_h = img_h
        self.img_w = img_w

        self.is_normalized = True
        if self.is_normalized:
            print('feature will be normalized')

    def compute(self, max_rank=5):
        self.q_pids, self.q_camids = Evaluator._get_info(self.dataset.query)
        self.g_pids, self.g_camids = Evaluator._get_info(self.dataset.gallery)

        print('start preparation')
        self._prepare_gallery_feats()
        print('shape of the gallery features: {}'.format(self.g_feats.shape))
        self._prepare_query_feats()
        print('shape of query features: {}'.format(self.q_feats.shape))

        self.distmat = Evaluator._compute_euclidean_distmat(self.g_feats, self.q_feats)
        # self.distmat = Evaluator._compute_cosine_distmat(self.g_feats, self.q_feats)
        print('distmat shape: {}'.format(self.distmat.shape))

        cmc, mAP = Evaluator._eval_func(self.distmat, self.q_pids, self.g_pids,
                        self.q_camids, self.g_camids, max_rank=max_rank)
        print('cmc: {}'.format(cmc))
        print('mAP: {}'.format(mAP))

    def compute2(self):
        pass

    @staticmethod
    def _get_info(datas):
        pids = []
        camids = []
        for item in datas:
            _, pid, camid = item
            pids.append(int(pid))
            camids.append(int(camid))
        return np.asarray(pids), np.asarray(camids)

    @staticmethod
    def _prepare_features(model, dataset, img_h, img_w, is_normalized, batch_size=32):
        N = len(dataset)
        batch_num = np.ceil(N / batch_size).astype(np.int32)
        feats = []
        idx = 0

        for _ in range(batch_num):
            tmp = []

            for _ in range(batch_size):
                path, _, _ = dataset[idx]
                target_size = (img_h, img_w)
                img = load_img_to_array(path, target_size)
                tmp.append(img)
                idx += 1
                if idx == N:
                    break

            tmp = imagenet_process(np.asarray(tmp))
            feat = model.predict(tmp)
            feats.extend(feat)

        feats = np.asarray(feats)
        if is_normalized:
            feats = normalize(feats)
        return feats

    def _prepare_gallery_feats(self):
        self.g_feats = Evaluator._prepare_features(self.model,
                            self.dataset.gallery, self.img_h, self.img_w, self.is_normalized)

    def _prepare_query_feats(self):
        self.q_feats = Evaluator._prepare_features(self.model,
                            self.dataset.query, self.img_h, self.img_w,  self.is_normalized)

    @staticmethod
    def _compute_euclidean_distmat(g_feats, q_feats):
        print('using eclidean distance for evaluation')
        mats = []
        for _q_feat in q_feats:
            _mat = np.linalg.norm(g_feats - _q_feat, axis=1, ord=2)
            mats.append(_mat)

        distmat = np.asarray(mats)
        distmat = np.square(mats)
        return distmat

    @staticmethod
    def _compute_cosine_distmat(g_feats, q_feats):
        print('using cosine distance for evaluation')
        mats = []
        g_feats_norm = np.linalg.norm(g_feats, axis=1)
        for _q_feat_norm in q_feats:
            dot_prod = np.dot(g_feats, _q_feat_norm)
            _q_feat_norm = np.linalg.norm(_q_feat_norm)
            cosdist = -1 * dot_prod / (_q_feat_norm * g_feats_norm)
            mats.append(cosdist)

        distmat = np.asarray(mats)
        distmat = np.square(mats)
        return distmat

    @staticmethod
    def _eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
        num_q, num_g = distmat.shape
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))

        # keep in mind that the distmat stores the distance between the query and the gallery
        # in such a format: the i-th row j-th column value in the matrix means how far the
        # i-th query image away from the j-th gallery image

        # get a sorted indices matrix, apply this mask can obtain a sorted version of X
        indices = np.argsort(distmat, axis=1)
        # get a matches indices matrix, which indicate a match between the query and gallery
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        # compute cmc curve for each query
        all_cmc = []
        all_AP = []
        num_valid_q = 0.  # number of valid query
        for q_idx in range(num_q):
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)

            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            orig_cmc = matches[q_idx][keep]
            if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.

            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

        all_cmc = np.asarray(all_cmc)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)
        return all_cmc, mAP

    def euclidean_distance(self, feats):
        feat_num = 64
        feat1 = K.tile(K.expand_dims(feats, axis=0), [feat_num, 1, 1])
        feat2 = K.tile(K.expand_dims(feats, axis=1), [1, feat_num, 1])
        delta = feat1 - feat2
        return K.sqrt(K.sum(K.square(delta), axis=2) + K.epsilon())


    def single_eval(self):
        """
        For testing the evaluator funcionality
        """
        print('\nsingle evaluation')
        # prepare the gallery
        self.g_pids, self.g_camids = Evaluator._get_info(self.dataset.gallery)
        self._prepare_gallery_feats()
        # pick one image from query

        # rand_idx = np.random.randint(0, len(self.dataset.query))
        # query = self.dataset.query[rand_idx]

        acc = 0
        total = 0

        for query in self.dataset.query:
            # print('query image info (path, pid, camid):\n{}'.format(query))
            # load the image, make batch
            img = load_img_to_array(query[0], (self.img_h, self.img_w))
            img = imagenet_process(img)
            img = np.expand_dims(img, 0)
            feat = self.model.predict(img)
            # calculate the distance matrix
            distmat = Evaluator._compute_euclidean_distmat(self.g_feats, feat)
            indices = np.argsort(distmat, axis=1)
            sorted_pids = self.g_pids[indices]
            sorted_cams = self.g_camids[indices]

            # don't exclude the one with same pid and same camid
            # if sorted_pids[0][0] == query[1]:
            #     acc += 1

            # exclude the one withe
            k = 0
            while k < sorted_pids.shape[1]:
                if sorted_pids[0][k] != query[1]:
                    print('[wrong result] {} should be {}, but found: {}'.format(
                        query[0], query[1], sorted_pids[0][k]))
                    break
                elif sorted_pids[0][k] == query[1] and sorted_cams[0][k] != query[2]:
                    acc += 1
                    break
                else:
                    k += 1

            total += 1

        print('top1 acc: {}'.format(acc / total))
