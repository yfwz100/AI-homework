# -*- coding: utf8 -*-

from __future__ import division, print_function

__author__ = 'zhi'

import numpy as np
from numpy import linalg

from lecture8.kmeans import k_means


def spectral_clustering(similarity_matrix, k):
    w, v = linalg.eig(
        similarity_matrix - np.diagflat(similarity_matrix.sum(axis=0)))
    centers, label = k_means(v[:, np.argsort(w)[-k:]], k)
    return label


if __name__ == '__main__':
    def main():
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
        similarity_matrix = np.zeros((len(data), len(data)))
        for i, x in enumerate(data):
            for j, y in enumerate(data):
                similarity_matrix[i, j] = similarity_matrix[j, i] = \
                    1 - abs(x[0] - y[0]) / abs(abs(x[0]) + abs(y[0]))
        label = spectral_clustering(similarity_matrix, 3)
        print('\n'.join(str(t) for t in zip(data, label)))

    main()
