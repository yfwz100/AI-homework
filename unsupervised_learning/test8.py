# -*- coding: utf8 -*-

__author__ = 'zhi'

import unittest

import numpy as np

from lecture8.em import gaussian_cluster
from lecture8.kmeans import k_means
from lecture8.spectral_clustering import spectral_clustering


data = [
    [0.5],
    [0.9],
    [20],
    [1],
    [19],
    [1.8],
    [1.2],
    [19.4],
    [21],
    [20.1],
    [19.5],
    [19],
    [22],
    [21.5],
    [100]
]


class Lecture8Tests(unittest.TestCase):
    def test_em(self):
        center, label = gaussian_cluster(np.matrix(data), 3)
        assert len(set(label)) == 3

    def test_k_means(self):
        center, label = k_means(np.matrix(data), 5)
        assert len(set(label)) == 5

    def test_spectral(self):
        similarity_matrix = np.zeros((len(data), len(data)))
        for i, x in enumerate(data):
            for j, y in enumerate(data):
                similarity_matrix[i, j] = similarity_matrix[j, i] = \
                    1 - abs(x[0] - y[0]) / abs(abs(x[0]) + abs(y[0]))
        label = spectral_clustering(similarity_matrix, 3)
        assert len(set(label)) == 3


if __name__ == '__main__':
    unittest.main()
