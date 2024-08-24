import unittest
from collections import Counter

import numpy as np
from sklearn.datasets import make_blobs

from ccluster.size import ConstrainedSpectralClustering


class TestConstrainedKMeans(unittest.TestCase):

    def test_ok(self):
        test_cases = [
            [40, 10, 50],
            [40., 10., 50.],
            [40, 10],
            [98, 1],
            [98],
            [40],
            [],
            None
        ]
        for i, test_case in enumerate(test_cases):
            x, y = make_blobs()
            k = 3
            predicted_labels = ConstrainedSpectralClustering(k, cluster_sizes=test_case).fit_predict(x)

            _, counts = np.unique(predicted_labels, return_counts=True)
            self.assertTrue(Counter(test_case) <= Counter(counts))

    def test_ko(self):
        test_cases = [
            [40, 10, 55],
            [40, 10, 52],
            [99],
            [-1],
            [1.1],
        ]
        for test_case in test_cases:
            x, y = make_blobs()
            k = 3
            with self.assertRaises(ValueError):
                ConstrainedSpectralClustering(k, cluster_sizes=test_case).fit_predict(x)


if __name__ == '__main__':
    unittest.main()
