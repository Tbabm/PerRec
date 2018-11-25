# encoding=utf-8

import unittest
from ..similarities import *
from .. import similarities as sims
from numpy.testing import assert_allclose

class TestSimilarities(unittest.TestCase):
    def setUp(self):
        self.X = [
            [1, 0, 1, 0],
            [1, 1, 0, 1],
            [0, 1, 0, 0]
        ]
        self.Y = [
            [1, 0, 1, 0],
            [1, 1, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 0, 0]
        ]

    def test_cosine_similarity(self):
        result = [
            [1         , 0.40824829, 0          , 0],
            [0.40824829, 1         , 0.57735027, 0],
            [0         , 0.57735027, 1         , 0]
        ]
        similarities = cosine_similarity(self.X, self.Y)
        assert_allclose(result, similarities)
    
    def test_euclidean_similarity(self):
        result = [
            [np.inf, 0.57735027, 0.57735027, 0.70710678],
            [0.57735027, np.inf, 0.70710678, 0.57735027,],
            [0.57735027, 0.70710678, np.inf, 1]
        ]
        similarities = euclidean_similarity(self.X, self.Y)
        assert_allclose(result, similarities)
    
    def test_jaccard_similarity(self):
        result = [
            [1, 0.25, 0, 0],
            [0.25, 1, 0.33333333, 0],
            [0, 0.33333333, 1, 0]
        ]
        similarities = jaccard_similarity(self.X, self.Y)
        assert_allclose(result, similarities)
    
    def test_correlation_similarity(self):
        # array([[1.        , 0.21132487, 0.21132487, 0.5       ],
        #        [0.21132487, 1.        , 0.66666667, 0.5       ],
        #        [0.21132487, 0.66666667, 1.        , 0.5       ]])
        result = sims._compute_similarities(correlation_similarity_1D, self.X, self.Y)
        result[:, -1] = np.array([0.5, 0.5, 0.5])
        X = np.array(self.X)
        Y = np.array(self.Y)
        similarities = correlation_similarity(X, Y)
        assert_allclose(result, similarities)

    def test_torch_cosine_similarity(self):
        a = torch.randn(3, 2).double()
        b = torch.randn(4, 2).double()
        sim1 = torch_cosine_similarity(a, b)
        sim2 = cosine_similarity(a.numpy(), b.numpy())
        assert_allclose(sim1, sim2)

    def test_calculate_doc_partial_similarities(self):
        # a = [torch.DoubleTensor([[-0.66280855, -0.40342563]]),
        #      torch.DoubleTensor([[ 1.42738917, -2.08969945],
        #                [-1.20685194, -0.19919694]])]
        # b = [torch.DoubleTensor([[-0.77491406, -0.39541403]]), 
        #      torch.DoubleTensor([[ 0.79141788,  0.14895806],
        #                [ 0.38944222, -1.21501918]]), 
        #      torch.DoubleTensor([[ 0.95401881,  0.7989287 ],
        #                [ 1.43606952, -0.20911604],
        #                [ 0.24571258, -2.08214752]])]
        # cosine similarities
        # [
        #   [
        #    tensor([[0.9972]], dtype=torch.float64), 
        #    tensor([[-0.9356,  0.2344]], dtype=torch.float64), 
        #    tensor([[-0.9887, -0.7704,  0.4162]], dtype=torch.float64)
        #   ], 
        #   [
        #    tensor([[-0.1271], [ 0.9529]], dtype=torch.float64), 
        #    tensor([[ 0.4016,  0.9585], [-0.9997, -0.1461]], dtype=torch.float64), 
        #    tensor([[-0.0977,  0.6771,  0.8862], [-0.8610, -0.9529,  0.0461]], dtype=torch.float64)
        #   ]
        # ]

        a = [np.array([[-0.66280855, -0.40342563]]),
             np.array([[ 1.42738917, -2.08969945],
                       [-1.20685194, -0.19919694]])]
        b = [np.array([[-0.77491406, -0.39541403]]), 
             np.array([[ 0.79141788,  0.14895806],
                       [ 0.38944222, -1.21501918]]), 
             np.array([[ 0.95401881,  0.7989287 ],
                       [ 1.43606952, -0.20911604],
                       [ 0.24571258, -2.08214752]])]
        c = [[1.0] * 1, [0.5] * 2]
        # doc_sims = [
        #     [0.9972, 0.2344, 0.4162],
        #     [0.4129, 0.4062, 0.46615]
        # ]
        doc_sims = [
            [0.99719368, 0.23438481, 0.41623218],
            [0.41288687, 0.40621558, 0.46613011]
        ]
        result = calculate_doc_partial_similarities(a, b, c, "cuda:0")
        assert_allclose(doc_sims, result)

if __name__ == "__main__":
    unittest.main()
