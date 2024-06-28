import unittest
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.datasets import make_blobs

from ccluster.size import constrained_k_means, ConstrainedKMeans


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
        for init in ['random', 'k-means++']:
            for i, test_case in enumerate(test_cases):
                x, y = make_blobs()
                k = 3
                centers, predicted_labels, _ = constrained_k_means(x, k,
                                                                   init=init,
                                                                   cluster_sizes=test_case)

                _, counts = np.unique(predicted_labels, return_counts=True)
                plt.scatter(x[:, 0], x[:, 1], c=predicted_labels, alpha=.5)
                plt.scatter(centers[:, 0], centers[:, 1], marker='x', color='k')
                plt.show()
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
                constrained_k_means(x, k, cluster_sizes=test_case)

    def test_sparse_ok(self):
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
        for init in ['random', 'k-means++']:
            for i, test_case in enumerate(test_cases):
                x, y = make_blobs()
                x_sparse = csr_matrix(x)
                k = 3
                centers, predicted_labels, _ = constrained_k_means(x_sparse, k,
                                                                   init=init,
                                                                   cluster_sizes=test_case)
                _, counts = np.unique(predicted_labels, return_counts=True)
                plt.scatter(x[:, 0], x[:, 1], c=predicted_labels, alpha=.5)
                plt.scatter(centers[:, 0], centers[:, 1], marker='x', color='k')
                plt.show()
                self.assertTrue(Counter(test_case) <= Counter(counts))

    def test_sparse_ko(self):
        test_cases = [
            [40, 10, 55],
            [40, 10, 52],
            [99],
            [98, 2, 0],
            [-1],
            [1.1],
        ]
        for test_case in test_cases:
            x, y = make_blobs()
            x = csr_matrix(x)
            k = 3
            with self.assertRaises(ValueError):
                constrained_k_means(x, k, cluster_sizes=test_case)

    def test_class_ok(self):
        test_cases = [
            [40, 10, 50],
            [40., 10., 50.],
            [40, 10],
            [98],
            [40],
            [],
            None
        ]
        for init in ['random', 'k-means++']:
            for i, test_case in enumerate(test_cases):
                x, y = make_blobs()
                x_sparse = csr_matrix(x)
                k = 3
                km = ConstrainedKMeans(n_clusters=k, init=init, cluster_sizes=test_case)
                predicted_labels = km.fit_predict(x_sparse)
                km.fit_transform(x_sparse)
                centers = km.cluster_centers_
                _, counts = np.unique(predicted_labels, return_counts=True)
                plt.scatter(x[:, 0], x[:, 1], c=predicted_labels, alpha=.5)
                plt.scatter(centers[:, 0], centers[:, 1], marker='x', color='k')
                plt.show()
                self.assertTrue(Counter(test_case) <= Counter(counts))

    def test_inertia_ok(self):
        test_cases = [
            [40, 10, 50],
            [40., 10., 50.],
            [40, 10],
            [98],
            [40],
            [],
            None
        ]
        for init in ['random', 'k-means++']:
            for i, test_case in enumerate(test_cases):
                x, y = make_blobs()
                x_sparse = csr_matrix(x)
                k = 3
                f_centers, f_predicted_labels, f_inertia = constrained_k_means(x_sparse, k,
                                                                               init=init,
                                                                               cluster_sizes=test_case,
                                                                               random_state=0)

                km = ConstrainedKMeans(n_clusters=k,
                                       init=init,
                                       cluster_sizes=test_case,
                                       random_state=0)
                predicted_labels = km.fit_predict(x_sparse)
                inertia = 0
                for c in range(k):
                    inertia += np.sum((x[predicted_labels == c] - km.cluster_centers_[c]) ** 2)

                self.assertAlmostEqual(km.inertia_, inertia)
                self.assertAlmostEqual(km.inertia_, f_inertia)

                inertia = 0
                for c in range(k):
                    inertia += np.sum((x[predicted_labels == c] - km.cluster_centers_[c]) ** 2)

                self.assertAlmostEqual(km.inertia_, inertia)
                self.assertAlmostEqual(km.inertia_, f_inertia)

    def test_predict(self):
        test_cases = [
            [40, 10, 50],
            [40., 10., 50.],
            [40, 10],
            [98],
            [40],
            [],
            None
        ]
        for init in ['random', 'k-means++']:
            for i, test_case in enumerate(test_cases):
                x, y = make_blobs()
                x_sparse = csr_matrix(x)
                k = 3

                km = ConstrainedKMeans(n_clusters=k,
                                       init=init,
                                       cluster_sizes=test_case,
                                       random_state=0)
                predicted_labels = km.fit_predict(x_sparse)

                f_predicted_labels = km.predict(x_sparse, cluster_sizes=test_case)
                self.assertTrue(np.all(predicted_labels == f_predicted_labels))


if __name__ == '__main__':
    unittest.main()
