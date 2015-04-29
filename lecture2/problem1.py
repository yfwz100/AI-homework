# -*- coding: utf8 -*-

from __future__ import print_function

__author__ = 'Wang Zhi'

"""
Problem 1: Given the map describe as the following

    route('A', 'Z', 75)
    route('A', 'T', 118)
    route('A', 'S', 140)
    route('B', 'P', 101)
    route('B', 'U', 85)
    route('B', 'G', 90)
    route('B', 'F', 211)
    route('C', 'D', 120)
    route('C', 'P', 138)
    route('C', 'R', 146)
    route('D', 'M', 75)
    route('E', 'H', 86)
    route('F', 'S', 99)
    route('G', 'B', 90)
    route('H', 'U', 98)
    route('I', 'N', 87)
    route('I', 'V', 92)
    route('L','M', 70)
    route('L', 'T', 111)
    route('O', 'Z', 71)
    route('O', 'S', 151)
    route('P', 'R', 97)
    route('P', 'B', 101)
    route('R', 'S', 80)
    route('U', 'V', 142)

and estimated cost to B: (in alphabetic order)

    366, 0, 160, 242, 161, 178, 77, 151, 226, 244, 241, 234, 380, 98, 193, 253, 329, 80, 199, 374

Find the best route from Z to B using Uniform cost search, greedy search and A* search.
"""

from graphsearch import GraphSearch


def main():
    symbol = ['Arad',
              'Bucharest',
              'Craiova',
              'Dobreta',
              'Eforie',
              'Fagaras',
              'Giurgiu',
              'Hirsova',
              'Iasi',
              'Lugoj',
              'Mehadia',
              'Neamt',
              'Oradea',
              'Pitesti',
              'Rimnicu Vilcea',
              'Sibiu',
              'Timisoara',
              'Urziceni',
              'Vaslui',
              'Zerind']

    cost_graph = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 140, 118, 0, 0, 75],
                  [0, 0, 0, 0, 0, 211, 90, 0, 0, 0, 0, 0, 0, 101, 0, 0, 0, 85, 0, 0],
                  [0, 0, 0, 120, 0, 0, 0, 0, 0, 0, 0, 0, 0, 138, 146, 0, 0, 0, 0, 0],
                  [0, 0, 120, 0, 0, 0, 0, 0, 0, 0, 75, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 86, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 211, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 99, 0, 0, 0, 0],
                  [0, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 86, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 98, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 87, 0, 0, 0, 0, 0, 0, 92, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 0, 0, 0, 0, 0, 111, 0, 0, 0],
                  [0, 0, 0, 75, 0, 0, 0, 0, 0, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 87, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 151, 0, 0, 0, 71],
                  [0, 101, 138, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 97, 0, 0, 0, 0, 0],
                  [0, 0, 146, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 97, 0, 80, 0, 0, 0, 0],
                  [140, 0, 0, 0, 0, 99, 0, 0, 0, 0, 0, 0, 151, 0, 80, 0, 0, 0, 0, 0],
                  [118, 0, 0, 0, 0, 0, 0, 0, 0, 111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 85, 0, 0, 0, 0, 0, 98, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 142, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 92, 0, 0, 0, 0, 0, 0, 0, 0, 142, 0, 0],
                  [75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 71, 0, 0, 0, 0, 0, 0, 0]]

    est_cost = [366, 0, 160, 242, 161, 178, 77, 151, 226, 244, 241, 234, 380, 98, 193, 253, 329, 80, 199, 374]

    graph_map = GraphSearch(cost_graph)

    print('Uniform:', '→'.join(map(lambda d: symbol[d],
                                   graph_map.uniform(symbol.index('Zerind'), symbol.index('Bucharest')).path)))
    print('Greedy:', '→'.join(map(lambda d: symbol[d],
                                  graph_map.greedy(symbol.index('Zerind'), symbol.index('Bucharest'), est_cost).path)))
    print('A*:', '→'.join(map(lambda d: symbol[d],
                              graph_map.a_star(symbol.index('Zerind'), symbol.index('Bucharest'), est_cost).path)))


if __name__ == '__main__':
    main()
