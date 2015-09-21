# -*- coding: utf8 -*-

from __future__ import print_function

import numpy as np
import heapq

__author__ = 'Wang Zhi'

__all__ = ['GraphSearch', 'SearchNode', 'print_search_tree']


class SearchNode(object):
    """
    The search node that back a search tree.
    This node class represent a node in a search tree and provides functionality to populate a search tree.
    """

    def __init__(self, name, distance=0, parent=None):
        """
        Initialize a search node by given name, distance and parent(maybe null if the node is the root).

        :param name: the name of the node.
        :param distance: the distance of the node.
        :param parent: the parent node.
        """
        super(SearchNode, self).__init__()
        self.__name = name
        self.__dist = distance
        self.__parent = parent
        if self.__parent is not None:
            assert isinstance(parent, SearchNode)
            self.__parent.__children.append(self)
        self.__children = []

    @property
    def parent(self):
        """
        Get the parent node.
        """
        return self.__parent

    @property
    def children(self):
        """
        Get the children of the node.
        :rtype: tuple
        :return: the tuple of the children.
        """
        return tuple(self.__children)

    @property
    def name(self):
        """
        Get the name of the node.
        """
        return self.__name

    @property
    def distance(self):
        """
        Get the overall distance of the node (from root to current node).
        """
        if self.parent is None:
            return self.__dist
        else:
            return self.parent.distance + self.__dist

    @property
    def path(self):
        """
        Get the path along the root.
        """
        p = [self.name]
        current_node = self
        while current_node.parent is not None:
            p.insert(0, current_node.parent.name)
            current_node = current_node.parent
        return p

    @property
    def root(self):
        """
        Get the root of the search tree that the node is in.
        """
        current_node = self
        while current_node.parent is not None:
            current_node = current_node.parent
        return current_node

    def __str__(self):
        return '→'.join([str(name) for name in self.path])


def print_search_tree(search_node, display=lambda d: d):
    """
    Print the search tree from the search_node.

    :param search_node: the search node.
    :type search_node: SearchNode
    :param display: the display function.
    """

    def _print(search_node, display, indent=0):
        print('┃' * (indent - 1) + '┣' * (1 if search_node.parent is not None else 0) + display(search_node.name))
        for child_node in search_node.children:
            _print(child_node, display, indent + 1)

    return _print(search_node, display)


class GraphSearch:
    """
    The graph search class that encapsulate a matrix-based graph.
    """

    class Node:
        """
        The node used by graph search.
        """

        def __init__(self, name, distance):
            """
            Initialize a node by given name and distance.

            :param name: the name of the node.
            :param distance: the distance of the node.
            :return: new node.
            """
            self.__name = name
            self.__dist = distance

        @property
        def name(self):
            return self.__name

        @property
        def distance(self):
            return self.__dist

    def __init__(self, graph):
        """
        Initialize a graph search object by given matrix-based graph.

        :param graph: the matrix-based graph.
        :return: new graph search object.
        """
        self.__graph = np.asarray(graph)

    def forward_nodes(self, node):
        """
        Find the forward nodes.

        :param node: the node name to find.
        :return: a generator of the found nodes.
        """
        for i in xrange(len(self.__graph[node])):
            if self.__graph[node, i] > 0:
                yield GraphSearch.Node(i, self.__graph[node, i])

    def backward_nodes(self, node):
        """
        Find the backward nodes.

        :param node: the node name to find.
        :return: a generator of the found nodes.
        """
        for i in xrange(len(self.__graph[:, node])):
            if self.__graph[i, node] > 0:
                yield GraphSearch.Node(i, self.__graph[node, i])

    def dfs(self, src, goal):
        """
        Depth first search.

        :param src: the name of the source.
        :param goal: the name of the goal.
        :return: the terminal node of the search tree.
        """
        node_stack = [SearchNode(src)]
        current_node = node_stack.pop()
        visited = {current_node.name}
        while current_node.name is not goal:
            for n in self.forward_nodes(current_node.name):
                if n.name not in visited:
                    node_stack.append(SearchNode(n.name, n.distance, current_node))
                    visited.add(n.name)
            current_node = node_stack.pop()
        return current_node

    def bfs(self, src, goal):
        """
        Breadth first search.

        :param src: the name of the source.
        :param goal: the name of the goal.
        :return: the terminal node of the search tree.
        """
        node_queue = [SearchNode(src)]
        current_node = node_queue.pop(0)
        visited = {current_node.name}
        while current_node.name is not goal:
            for n in self.forward_nodes(current_node.name):
                if n.name not in visited:
                    node_queue.append(SearchNode(n.name, n.distance, current_node))
                    visited.add(n.name)
            current_node = node_queue.pop(0)
        return current_node

    def uniform(self, src, goal):
        """
        Uniform cost search for a route from src to goal.

        :param src: the name of the source.
        :param goal: the name of the goal.
        :return: the terminal node of the search tree.
        """
        node_heap = [(0, SearchNode(src))]
        current_node = heapq.heappop(node_heap)
        while current_node[1].name is not goal:
            for n in self.forward_nodes(current_node[1].name):
                heapq.heappush(node_heap,
                               (current_node[1].distance + n.distance, SearchNode(n.name, n.distance, current_node[1])))
            current_node = heapq.heappop(node_heap)
        return current_node[1]

    def greedy(self, src, goal, est_cost):
        """
        Greedy search for a route from src to goal.

        :param src: the name of the source.
        :param goal: the name of the goal.
        :param est_cost: the array of estimated cost to the goal.
        :return: the terminal node of the search tree.
        """
        node_heap = [(est_cost[src], SearchNode(src))]
        current_node = heapq.heappop(node_heap)
        while current_node[1].name is not goal:
            for n in self.forward_nodes(current_node[1].name):
                heapq.heappush(node_heap, (est_cost[src], SearchNode(n.name, n.distance, current_node[1])))
            current_node = heapq.heappop(node_heap)
        return current_node[1]

    def a_star(self, src, goal, est_cost):
        """
        A* search for a (optimal) route from src to goal.

        :param src: the name of the source.
        :param goal: the name of the goal.
        :param est_cost: the array of estimated cost to the goal.
        :return: the terminal node of the search tree.
        """
        node_heap = [(est_cost[src], SearchNode(src))]
        current_node = heapq.heappop(node_heap)
        while current_node[1].name is not goal:
            for n in self.forward_nodes(current_node[1].name):
                heapq.heappush(node_heap,
                               (current_node[1].distance + n.distance + est_cost[n.name],
                                SearchNode(n.name, n.distance, current_node[1])))
            current_node = heapq.heappop(node_heap)
        return current_node[1]

    @property
    def graph(self):
        """
        Get the graph matrix.

        :return: the copy of the graph matrix.
        """
        return self.__graph.copy()

