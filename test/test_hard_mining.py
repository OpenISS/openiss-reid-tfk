import unittest
import numpy as np
import keras.backend as K
from tools import distance as dist


class TestHardMining(unittest.TestCase):
    def setUp(self):
        # manumally create a 2d matrix
        self.distmat = np.array([
            [0,  98,   111, 149, 141, 158, 100, 128, 146],
            [98,   0,  156, 143, 169, 125, 102, 137, 152],
            [111, 156,   0, 138,  92, 142, 137, 123, 124],
            [149, 143, 138,   0, 163, 122, 110, 123, 118],
            [141, 169,  92, 163,   0, 128, 146, 137, 144],
            [158, 125, 142, 122, 128,   0, 119, 139, 134],
            [100, 102, 137, 110, 146, 119,   0, 123, 132],
            [128, 137, 123, 123, 137, 139, 123,   0,  49],
            [146, 152, 124, 118, 144, 134, 132,  49,   0],
        ])
        self.label = np.asarray([1, 1, 1, 2, 2, 2, 3, 3, 3])

    def test_hard_sample_mining(self):
        expected = np.array([
            [14], [57], [67], [56], [74], [12], [35], [3], [17]
        ])
        ap, an = dist.hard_sample_mining(self.distmat, self.label)
        sub = 3 + ap - an
        # print(K.eval(sub))
        self.assertTrue(np.all([expected, K.eval(sub)]))


if __name__ == '__main__':
    unittest.main()
