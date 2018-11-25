# encoding=utf-8

import numpy as np
import unittest
from numpy.testing import assert_array_almost_equal

from ..nearest_neighbor import reserve_nn_scores, cal_perm_scores

class TestNearestNeighbor(unittest.TestCase):
    def setUp(self):
        self.scores = np.array([
                           [0.8180, 0.3675, 0.0862, 0.2476],
                           [0.6461, 0.3863, 0.8757, 0.0605],
                           [0.3610, 0.5352, 0.7467, 0.0101]
                          ])
        self.reserve_nn_scores = np.array([
                           [0.8180, 0.3675, 0.0   , 0.0],
                           [0.6461, 0.0   , 0.8757, 0.0],
                           [0.0   , 0.5352, 0.7467, 0.0]
                          ])
        self.train_perm_vectors = np.array([
                           [1, 0, 0, 0, 1],
                           [0, 1, 0, 0, 1],
                           [1, 0, 1, 0, 0],
                           [0, 1, 0, 1, 0]   
                          ])
        self.perm_scores = np.array([
                           [0.8180, 0.3675, 0     , 0    , 1.1855],
                           [1.5218, 0     , 0.8757, 0    , 0.6461],
                           [0.7467, 0.5352, 0.7467, 0    , 0.5352]
                          ])

    def test_reserve_nn_scores(self):
        new_scores = reserve_nn_scores(self.scores, nn_num=2)
        assert_array_almost_equal(self.reserve_nn_scores, new_scores)

    def test_cal_perm_scores(self):
        perm_scores = cal_perm_scores(self.reserve_nn_scores, self.train_perm_vectors)
        assert_array_almost_equal(self.perm_scores, perm_scores)

if __name__ == "__main__":
    unittest.main()
