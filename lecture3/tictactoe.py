# -*- coding: utf8 -*-

"""
The TicTacToe game.

This is the game to describe the AI of the TicTacToe game using Minimax strategy. The board of the game is described by
a 3x3 matrix of numbers. '0' means empty and the a pair of reciprocal number is used to describe the players in the
game. The two functions defined below follows the convention.
"""

__author__ = 'yfwz100'

import numpy as np


def agent_utility(board, pos, mark, step=1):
    """
    Get the utility of the agent given the current board and position.

    :param board: the board of the game, a 3x3 matrix.
    :param pos: the position to play.
    :param mark: the mark of the player.
    :param step: steps to consider.
    :return: the utility tuple.
    """
    if board[pos] == 0:
        board = board.copy()
        board[pos] = mark
        if step == 1:
            return (
                # the chances for opponent to win.
                -sum(
                    sum(1 if np.count_nonzero((r == 0) | (r == -mark)) == 3 else 0 for r in m)
                    for m in [board, board.T, board.flat[[[0, 4, 8], [2, 4, 6]]]]
                ),
                # the chances for self to win.
                sum(
                    sum(1 if np.count_nonzero((r == 0) | (r == mark)) == 3 else 0 for r in m)
                    for m in [board, board.T, board.flat[[[0, 4, 8], [2, 4, 6]]]]
                ),
                # the steps for opponent to win.
                min(
                    min(3 - np.count_nonzero(r == -mark) if np.count_nonzero((r == 0) | (r == -mark)) == 3 else 3
                        for r in m)
                    for m in [board, board.T, board.flat[[[0, 4, 8], [2, 4, 6]]]]
                )
            )
        else:
            u, oppo = agent_choice(board, -mark, step - 1)
            board[oppo] = -mark
            u, p = agent_choice(board, mark, step - 1)
            return u
    else:
        raise RuntimeError('Not possible.')


def agent_position_range(board, mark):
    """
    Find the potential position to reduce the runtime overhead.

    :param board: the board of the game.
    :param mark: the mark of the player.
    :return: the position list.
    """
    return np.transpose(np.nonzero(board == 0))


def agent_choice(board, mark, step=1):
    """
    Get the choice of the agent given the board and mark.

    :param board: the board of the game, a 3x3 matrix.
    :param mark: the mark of the player.
    :param step: number of steps to consider.
    :return: the position of next step.
    """
    return max(((agent_utility(board, tuple(p), mark, step), tuple(p)) for p in agent_position_range(board, mark)),
               key=lambda d: d[0])
