# -*- coding: utf8 -*-

from __future__ import print_function

__author__ = 'Wang Zhi'

import numpy as np
import abc


class Strategy(object):
    """
    The base class for strategy used in the following methods.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def choice(self):
        pass

    @abc.abstractproperty
    def utility(self):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass


def single_move(u_matrix, player):
    """
    Single move strategy of the Minimax strategy.

    :param u_matrix: the utility matrix(2x2x2)
    :param player: the player ID (numeric)
    :return: the strategy
    """

    class SMStrategy(Strategy):
        def __init__(self, choice, utility):
            self.__choice = choice
            self.__utility = utility

        @property
        def choice(self):
            return self.__choice

        @property
        def utility(self):
            return self.__utility

        def __str__(self):
            return "[choice: %s, utility: %s]" % (self.choice, self.utility)

    strategies = [
        min(u_matrix.take(i, axis=player).take(player, axis=1)) for i in xrange(2)
    ]
    return SMStrategy(np.argmax(strategies), max(strategies))


def prob_move(u_matrix, player):
    """
    Probability distribution move in Minimax strategy.

    :param u_matrix: the utility matrix (2x2x2)
    :param player: the player ID (numeric)
    :return: the probability strategy
    """

    class ProbStrategy(Strategy):

        def __init__(self, choice, utility):
            self.__choice = choice
            self.__utility = utility

        @property
        def choice(self):
            return self.__choice

        @property
        def utility(self):
            return self.__utility

        def __str__(self):
            return "[choice: %.2f of choice 0, utility: %s]" % (self.choice, self.utility)

    param = [
        u_matrix.take(i, axis=player).take(player, axis=1) for i in xrange(2)
    ]

    if param[0][0] - param[0][1] + param[1][1] - param[1][0] != 0:
        p = (param[1][1] - param[1][0]) / float(param[0][0] - param[0][1] + param[1][1] - param[1][0])
        u = param[0][0] * p - param[0][1] * (1 - p)
        return ProbStrategy(p, u)
    else:
        # fallback to the minimax strategy.
        return single_move(u_matrix, player)


def problem4():
    """
    The utility matrix given as following:

        [[5, -5], [3, -3]],
        [[4, -4], [2, -2]]

    Calculate their utility when they follow the best strategy.
    """

    utility_matrix = np.array([
        [[5, -5], [3, -3]],
        [[4, -4], [2, -2]]
    ])

    A = 0
    B = 1

    print('Single move strategy:')
    print('  Player A:', single_move(utility_matrix, A))
    print('  Player B:', single_move(utility_matrix, B))

    print('Probability distribution on move:')
    print('  Player A:', prob_move(utility_matrix, A))
    print('  Player B:', prob_move(utility_matrix, B))


def problem5():
    """
    The utility matrix given as following:

        [[3, -3], [6, -6]],
        [[5, -5], [4, -4]]

    Calculate their utility when they follow the best strategy.
    """

    utility_matrix = np.array([
        [[3, -3], [6, -6]],
        [[5, -5], [4, -4]]
    ])

    A = 0
    B = 1

    print('Single move strategy:')
    print('  Player A:', single_move(utility_matrix, A))
    print('  Player B:', single_move(utility_matrix, B))

    print('Probability distribution on move:')
    print('  Player A:', prob_move(utility_matrix, A))
    print('  Player B:', prob_move(utility_matrix, B))


if __name__ == '__main__':
    print('Problem 4:')
    problem4()

    print()

    print('Problem 5:')
    problem5()