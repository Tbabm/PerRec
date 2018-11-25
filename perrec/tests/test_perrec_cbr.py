# encoding=utf-8

import unittest

import numpy as np
from numpy.testing import assert_array_equal
from scipy.sparse.csr import csr_matrix

from ..common.dataset import load_dataset, Dataset
from ..cbr import PerRecCBR

class TestPerRecCBR(unittest.TestCase):
    def test_build_perm_docs(self):
        perm_vectors = csr_matrix([
            [1, 0, 1],
            [0, 1, 1]
        ])
        api_vectors = csr_matrix([
            [1,0],
            [0,1]
        ])
        perm_docs = np.array([
            [1, 0],
            [0, 1],
            [1, 1]
        ])
        result = PerRecCBR.build_perm_docs(perm_vectors, api_vectors)
        assert_array_equal(perm_docs, result)
    
    def test_perrec_cbr(self):
        np.warnings.filterwarnings('ignore')
        dataset = load_dataset()
        cur_dataset = dataset[:5]
        estimator = PerRecCBR(sim_func="cosine")
        api_lists = Dataset.extract_app_api_lists(cur_dataset)
        perm_lists = Dataset.extract_app_perm_lists(cur_dataset)
        estimator.fit(api_lists, perm_lists)
        test_api_lists = Dataset.extract_app_api_lists(dataset[5:6])
        results = estimator.transform(test_api_lists)
        print(results)

if __name__ == "__main__":
    unittest.main()
