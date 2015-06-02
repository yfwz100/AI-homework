# -*- coding: utf8 -*-

from __future__ import division, print_function

__author__ = 'zhi'

import itertools
import numpy as np
import random


def k_means(data, k, init_center=None, max_iter=10, dist=lambda x, y: np.dot((x - y), (x - y).T), tau=1e-6):
    old_center = None
    if init_center is None:
        center = random.sample([np.asmatrix(d) for d in data], k)
    else:
        center = [np.asmatrix(c) for c in init_center]
    i = 0
    label = [
        np.argmax([dist(c, x) for c in center]) for x in data
    ]
    while i < max_iter and \
            (old_center is None or not all(np.allclose(o, c, tau) for o, c in zip(old_center, center))):
        old_center = center
        # find the mean of the cluster.
        center = [
            np.average([c[0] for c in cluster], axis=0)
            for label, cluster in itertools.groupby(sorted(zip(data, label),
                                                           key=lambda d: d[1]),
                                                    key=lambda d: d[1])
        ]
        # label the data.
        label = [
            np.argmin([dist(c, x) for c in center]) for x in data
        ]
        # increment iteration.
        i += 1
    return center, label


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
            [21.5]
        ]
        center, label = k_means(np.matrix(data), 2)
        print(center)
        print('\n'.join(str(t) for t in zip(data, label)))

    main()