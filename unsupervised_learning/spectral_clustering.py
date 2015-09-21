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
