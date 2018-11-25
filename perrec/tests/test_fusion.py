# encoding=utf-8

import unittest

from ..fusion import *

from numpy.testing.utils import assert_array_equal, assert_allclose

class TestFusion(unittest.TestCase):
    def setUp(self):
        self.score_sets = np.array(
            [[[-0.1611121 , -0.24623327, 0,  0.86233683],
            [-0.06174189, 0,  0.30196532,  0.97658718],
            [ 0.76403641,  1.83459046,  1.11651605,  0.34346855]],

            [[-0.43638031, -0.13377434,  0.1681035 ,  0.16403972],
            [ 1.0423727 , 0, 0,  1.47440706],
            [-0.61129717,  1.37992526,  1.19066629, -0.68834553]]])

    def test_sum_fusion(self):
        new_scores = sum_fusion(self.score_sets)
        result = np.array([[-0.59749241, -0.38000761,  0.1681035 ,  1.02637655],
                        [ 0.98063081,  0.        ,  0.30196532,  2.45099424],
                        [ 0.15273924,  3.21451572,  2.30718234, -0.34487698]])
        assert_allclose(result, new_scores)
    
    def test_anz_fusion(self):
        new_scores = anz_fusion(self.score_sets)
        result = np.array([[-0.2987462 , -0.1900038 ,  0.1681035 ,  0.51318828],
                            [ 0.49031541,         0,  0.30196532,  1.22549712],
                            [ 0.07636962,  1.60725786,  1.15359117, -0.17243849]])
        assert_allclose(result, new_scores)

    def test_mnz_fusion(self):
        new_scores = mnz_fusion(self.score_sets)
        result = np.array([[-1.19498482, -0.76001522,  0.1681035 ,  2.0527531 ],
                           [ 1.96126162,  0.        ,  0.30196532,  4.90198848],
                           [ 0.30547848,  6.42903144,  4.61436468, -0.68975396]])
        assert_allclose(result, new_scores)

    def test_max_fusion(self):
        new_scores = max_fusion(self.score_sets)
        result = np.array([[-0.1611121 , -0.13377434,  0.1681035 ,  0.86233683],
                           [ 1.0423727 ,  0.        ,  0.30196532,  1.47440706],
                           [ 0.76403641,  1.83459046,  1.19066629,  0.34346855]])
        assert_allclose(result, new_scores)

    def test_min_fusion(self):
        new_scores = min_fusion(self.score_sets)
        result = np.array([[-0.43638031, -0.24623327,  0.        ,  0.16403972],
                           [-0.06174189,  0.        ,  0.        ,  0.97658718],
                           [-0.61129717,  1.37992526,  1.11651605, -0.68834553]])
        assert_allclose(result, new_scores)

    def test_borda_count_fusion(self):
        new_scores = borda_count_fusion(self.score_sets)
        # [[[1 , 0, 2,  3],
        # [0, 1,  2,  3],
        # [ 1,  3,  2,  0]],

        # [[0, 1,  3 ,  2],
        # [ 2 , 0, 1,  3],
        # [1,  3,  2, 0]]]
        result = np.array([[1, 1, 5, 5],
                           [2, 1, 3, 6],
                           [2, 6, 4, 0]
        ])
        assert_allclose(result, new_scores)

    def test_fusion(self):
        perm_list = ['p1', 'p2', 'p3', 'p4']
        ranked_perms = fusion(self.score_sets, perm_list, 'borda_count')
        result = np.array([
            ['p3', 'p4', 'p1', 'p2'],
            ['p4', 'p3', 'p1', 'p2'],
            ['p2', 'p3', 'p1', 'p4']
        ])
        assert_array_equal(result, ranked_perms)

if __name__ == "__main__":
    unittest.main()