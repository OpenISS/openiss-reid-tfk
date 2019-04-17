import unittest
import numpy as np
import keras.backend as K
from tools import distance as dist


class TestDistance(unittest.TestCase):
    def setUp(self):
        # manumally create a 2d matrix
        self.data1 = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10]
        ])
        self.data2 = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10]
        ])
        # self.data2 = np.tile(np.arange(6, 11), (3,)).reshape(3, 5)
        print('test data\n{}\n{}'.format(self.data1, self.data2))

    def test_euclidean_distance(self):
        expected = np.array([[3.1622776e-04, 1.1180341e+01],
                             [1.1180341e+01, 3.1622776e-04]])
        A = K.constant(self.data1)
        B = K.constant(self.data2)
        e_dist = dist.euclidean_distance(A, B)
        self.assertTrue(np.all([expected, K.eval(e_dist)]))


if __name__ == '__main__':
    unittest.main()
