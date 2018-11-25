# encoding=utf-8

import unittest
import numpy as np

from ..scorers import *

class TestScorers(unittest.TestCase):
    def test_average_precision(self):
        y_true = ["perm2", "perm4", "perm8"]
        y_pred = ["perm2", "perm3", "perm4", "perm5"]
        result = 0.5556
        ap = round(average_precision(y_true, y_pred), 4)
        self.assertEqual(result, ap)
    
    def test_mean_average_precision(self):
        y_true = [
            ["perm2", "perm4", "perm8"],
            ["perm3"],
            ["perm5", "perm6", "perm9", "perm11", "perm12"]
        ]
        y_pred = [
            # 0.555555556
            ["perm2", "perm3", "perm4", "perm5"],
            # 0
            ["perm2", "perm4", "perm5", "perm6"],
            # (0.5 + 2/3 + 3/4) / 5 = 0.3833333
            ["perm2", "perm9", "perm11", "perm6"]
        ]
        mean_ap = round(mean_average_precision(y_true, y_pred),4)
        result = 0.3130
        self.assertEqual(result, mean_ap)

    def test_total_recall_ratio(self):
        candidate_num = 45
        cases = [
            (
                [],
                ["p1", "p2"]
            ),
            (
                ["p1", "p2"],
                []
            ),
            (
                ["p1"],
                ["p1", "p2"]
            ),
            (
                ["p1", "p2", "p3"],
                ["p4", "p3"]
            ),
            (
                np.array(["p1", "p2", "p3"]),
                np.array(["p4", "p3", "p1", "p5", "p2"])
            ),
        ]
        results = [1, 45/2, 1, 45/3, 5/3]
        for case, res in zip(cases, results):
            cur = total_recall_ratio(case[0], case[1], candidate_num)
            self.assertEqual(res, cur)

    def test_average_total_recall_ratio(self):
        candidate_num = 45
        y_true = [
            [],
            ["p1", "p2"],
            ["p1"],
            ["p1", "p2", "p3"],
            ["p1", "p2", "p3"]
        ]
        y_pred = [
            ["p1", "p2"],
            [],
            ["p1", "p2"],
            ["p4", "p3"],
            ["p4", "p3", "p1", "p5", "p2"]
        ]
        result = (39.5 + 5/3)/5
        cur = average_total_recall_ratio(y_true, y_pred, candidate_num)
        self.assertEqual(result, cur)

if __name__ == "__main__":
    unittest.main()
