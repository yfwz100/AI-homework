# -*- coding: utf8 -*-

from pprint import pprint
import random

import numpy
from numpy import linalg


__author__ = 'yfwz100'


def gaussian_cluster(data, k, init_center=None, tau=1e-3):
    old_center = None

    if init_center is None:
        center = random.sample([d for d in data], k)  # [d for d in data[-k:]]
    else:
        center = init_center
    pprint(center)
    sigma = [numpy.asmatrix(numpy.eye(data.shape[1]))] * k
    ez = numpy.ones((k, data.shape[0]))

    def norm_pdf(x, c, sig):
        d = linalg.det(sig)
        if d == 0:
            return 1 if numpy.allclose(x, c, 1e-6) else 1e-9
        else:
            return float(numpy.exp(-0.5 * (x - c) * (sig ** -1) * (x - c).T + 1e-6))

    while old_center is None or not all(numpy.allclose(o, c, tau) for o, c in zip(old_center, center)):
        old_center = center
        # e-step
        ez = numpy.array([[norm_pdf(x, c, sig) for x in data] for c, sig in zip(center, sigma)])
        ez = ez / ez.sum(axis=0)
        # m-step
        center = [numpy.average(data, axis=0, weights=w) for w in ez]
        sigma = [
            numpy.asmatrix(numpy.average([(x - c).T * (x - c) for x in data], axis=0, weights=w))
            for w, c in zip(ez, center)
            ]
        print("Center:")
        pprint(center)
        print("Sigma:")
        pprint(sigma)
        print("EZ:")
        pprint(ez)
    return center, [numpy.argmax(p) for p in zip(*ez)]
