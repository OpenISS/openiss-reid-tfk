# -* - coding: UTF-8 -* -

import numpy as np
import tensorflow as tf
import keras
from keras import backend as K



def triplet_loss(num_ids, num_imgs, margin):
    p = num_ids
    k = num_imgs

    # construct labels
    mask = [i for i in range(1, p + 1) for j in range(k)]
    mask = np.asarray(mask)
    feat_num = p * k

    def euclidean_distance(feats):
        feat1 = K.tile(K.expand_dims(feats,axis = 0), [feat_num,1,1])
        feat2 = K.tile(K.expand_dims(feats,axis = 1), [1,feat_num,1])
        delta = feat1 - feat2
        return K.sqrt(K.sum(K.square(delta),axis = 2) + K.epsilon())

    def hard_sample_mining(dist_mat, labels):
        assert len(dist_mat.shape) == 2
        assert dist_mat.shape[0] == dist_mat.shape[1]
        N = dist_mat.shape[0]

        L = K.reshape(K.tile(labels, [N]), [N, N])
        Lt = K.transpose(L)

        # is_pos and is_neg are mask matrices with size [N, N]
        # they can be the mask to indicate which pair is positive (negative)
        is_pos = K.equal(L, Lt)
        is_neg = K.not_equal(L, Lt)

        tmp_p = K.reshape(tf.boolean_mask(dist_mat, is_pos), [N, -1])
        tmp_n = K.reshape(tf.boolean_mask(dist_mat, is_neg), [N, -1])

        dis_ap = K.max(tmp_p, axis=1, keepdims=True)
        dis_an = K.min(tmp_n, axis=1, keepdims=True)

        return dis_ap, dis_an

    def triplet_hard_loss(y_true, y_pred):
        dist_mat = euclidean_distance(y_pred)
        dist_ap, dist_an = hard_sample_mining(dist_mat, mask)
        return K.mean(K.maximum(K.epsilon(), K.sum(margin + dist_ap - dist_an)))

    def triplet_all_loss(y_true, y_pred):
        N = dist_mat.shape[0]
        L = K.reshape(K.tile(labels, [N]), [N, N])
        Lt = K.transpose(L)

        dist_mat = euclidean_distance(y_pred)
        is_pos = K.equal(L, Lt)
        is_neg = K.not_equal(L, Lt)
        tmp_p = K.reshape(tf.boolean_mask(dist_mat, is_pos), [N, -1])
        tmp_n = K.reshape(tf.boolean_mask(dist_mat, is_neg), [N, -1])

        K.sum(tmp_p)

    return triplet_hard_loss

