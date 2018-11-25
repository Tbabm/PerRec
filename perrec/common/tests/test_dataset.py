# encoding=utf-8

import unittest

from perrec.common.dataset import Dataset

class TestDataset(unittest.TestCase):
    def setUp(self):
        # build fake apps
        app_dict = {
            'app1': {
                'apis': ['api1', 'api3', 'api5'],
                'perms': ['perm2', 'perm4', 'perm6'],
            },
            'app2': {
                'apis': ['api2', 'api3', 'api8'],
                'perms': ['perm1', 'perm2', 'perm5'],
            },
            'app3': {
                'apis': ['api1', 'api4', 'api5'],
                'perms': ['perm3', 'perm2', 'perm1'],
            }
        }
        self.dataset = Dataset(app_dict)

    def test_extract_perm_set(self):
        perm_lists = self.dataset.extract_perm_lists()
        perm_set = Dataset.extract_perm_set(perm_lists)
        result = ['perm1', 'perm2', 'perm3', 'perm4', 'perm5', 'perm6']
        self.assertEqual(result, perm_set)

if __name__ == "__main__":
    unittest.main()
