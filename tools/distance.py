import tensorflow as tf
import keras.backend as K

# convenience l2_norm function
def l2_norm(x, axis=None):
    """
    takes an input tensor and returns the l2 norm along specified axis
    """

    square_sum = K.sum(K.square(x), axis=axis, keepdims=True)
    norm = K.sqrt(K.maximum(square_sum, K.epsilon()))

    return norm


def pairwise_cosine_sim(A_B):
    """
    Arguments
        A [batch x n x d] tensor of n rows with d dimensions
        B [batch x m x d] tensor of n rows with d dimensions

    Returns
        D [batch x n x m] tensor of cosine similarity scores between each point i<n, j<m
    """

    A, B = A_B
    A_mag = l2_norm(A, axis=2)
    B_mag = l2_norm(B, axis=2)
    num = K.batch_dot(A_tensor, K.permute_dimensions(B_tensor, (0, 2, 1)))
    den = (A_mag * K.permute_dimensions(B_mag, (0, 2, 1)))
    dist_mat = num / den

    return dist_mat

def euclidean_distance(A, B):
    a = A.shape[0]
    b = B.shape[0]
    feat1 = K.tile(K.expand_dims(A, axis=0), [a, 1, 1])
    feat2 = K.tile(K.expand_dims(B, axis=1), [1, b, 1])
    delta = feat1 - feat2
    return K.sqrt(K.sum(K.square(delta), axis=2) + K.epsilon())

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
