# -*- coding: utf-8 -*-
__all__ = ['simu_graph', 'draw_graph', 'simu_grid_graph', 'HeadTailWrapper', 'head_proj', 'tail_proj',
           'minimal_spanning_tree', 'random_walk', 'fast_pcst']

import random
import numpy as np

try:
    import c_sparse_learn

    try:
        from c_sparse_learn import c_proj_pcst
        from c_sparse_learn import c_proj_head
        from c_sparse_learn import c_proj_tail
    except ImportError:
        print('cannot some functions')
        exit(0)
except ImportError:
    print('cannot find the package c_sparse_learn')


def simu_graph(num_nodes, rand=False, graph_type='grid'):
    """
    To generate a grid graph. Each node has 4-neighbors.
    :param num_nodes: number of nodes in the graph.
    :param rand: if rand True, then generate random weights in (0., 1.)
    :param graph_type: ['grid', 'chain']
    :return: edges and corresponding to unite weights.
    """
    edges, weights = [], []
    if graph_type == 'grid':
        length = int(np.sqrt(num_nodes))
        width, index = length, 0
        for i in range(length):
            for j in range(width):
                if (index % length) != (length - 1):
                    edges.append((index, index + 1))
                    if index + length < int(width * length):
                        edges.append((index, index + length))
                else:
                    if index + length < int(width * length):
                        edges.append((index, index + length))
                index += 1
        edges = np.asarray(edges, dtype=int)
    elif graph_type == 'chain':
        for i in range(num_nodes - 1):
            edges.append((i, i + 1))
    else:
        edges = []

    # generate weights of the graph
    if rand:
        weights = []
        while len(weights) < len(edges):
            rand_x = np.random.random()
            if rand_x > 0.:
                weights.append(rand_x)
        weights = np.asarray(weights, dtype=np.float64)
    else:
        weights = np.ones(len(edges), dtype=np.float64)
    return edges, weights


def simu_grid_graph(width, height, rand_weight=False):
    """Generate a grid graph.
    To generate a grid graph. Each node has 4-neighbors. Please see more
    details in https://en.wikipedia.org/wiki/Lattice_graph. For example,
    we can generate 5x3(width x height) grid graph
                0---1---2---3---4
                |   |   |   |   |
                5---6---7---8---9
                |   |   |   |   |
                10--11--12--13--14
    by using simu_grid_graph(5, 3)
    We can also generate a 1x5 chain graph
                0---1---2---3---4
    by using simu_grid_graph(5, 1)
    :param width: width of this grid graph.
    :param height: height of this grid graph.
    :param rand_weight: generate weights from U(1., 2.) if it is True.
    :return: edges and corresponding edge costs.
    return two empty [],[] list if there was any error occurring.
    """
    if width < 0 and height < 0:
        print('Error: width and height should be positive.')
        return [], []
    width, height = int(width), int(height)
    edges, weights = [], []
    index = 0
    for i in range(height):
        for j in range(width):
            if (index % width) != (width - 1):
                edges.append((index, index + 1))
                if index + width < int(width * height):
                    edges.append((index, index + width))
            else:
                if index + width < int(width * height):
                    edges.append((index, index + width))
            index += 1
    edges = np.asarray(edges, dtype=int)
    # random generate costs of the graph
    if rand_weight:
        weights = []
        while len(weights) < len(edges):
            weights.append(random.uniform(1., 2.0))
        weights = np.asarray(weights, dtype=np.float64)
    else:  # set unit weights for edge costs.
        weights = np.ones(len(edges), dtype=np.float64)
    return edges, weights


def random_walk(edges, s, init_node=None, restart=0.0):
    """Random generate a connected subgraph by using random walk.
    Given a connected undirected graph (represented as @param:edges), a random
    walk is a procedure to generate a connected subgraph with s different
    nodes. Please check more details in the first paragraph of section 1.
    basic notations and facts of reference [1] in Page 3.
    Reference:  [1] Lovász, László. "Random walks on graphs: A survey."
                    Combinatorics, Paul erdos is eighty 2.1 (1993): 1-46.
    :param edges: input graph as the list of edges.
    :param s: the number of nodes in the returned subgraph.
    :param init_node: initial point of the random walk.
    :param restart: with a fix probability to restart from the initial node.
    :return: a list of s nodes and a list of walked edges.
    return two empty list if there was any error occurring.
    """
    adj, nodes = dict(), set()
    for edge in edges:  # construct the adjacency matrix.
        uu, vv = int(edge[0]), int(edge[1])
        nodes.add(uu)
        nodes.add(vv)
        if uu not in adj:
            adj[uu] = set()
        adj[uu].add(vv)
        if vv not in adj:
            adj[vv] = set()
        adj[vv].add(uu)
    if init_node is None:
        # random select an initial node.
        rand_start_point = random.choice(list(nodes))
        init_node = list(adj.keys())[rand_start_point]
    if init_node not in nodes:
        print('Error: the initial_node is not in the graph!')
        return [], []
    if not (0.0 <= restart < 1.0):
        print('Error: the restart probability not in (0.0,1.0)')
        return [], []
    if not (0 <= s <= len(nodes)):
        print('Error: the number of nodes not in [0,%d]' % len(nodes))
        return [], []
    subgraph_nodes, subgraph_edges = set(), set()
    next_node = init_node
    subgraph_nodes.add(init_node)
    if s <= 1:
        return subgraph_nodes, subgraph_edges
    # get a connected subgraph with s nodes.
    while len(subgraph_nodes) < s:
        next_neighbors = list(adj[next_node])
        rand_nei = random.choice(next_neighbors)
        subgraph_nodes.add(rand_nei)
        subgraph_edges.add((next_node, rand_nei))
        subgraph_edges.add((rand_nei, next_node))
        next_node = rand_nei  # go to next node.
        if random.random() < restart:
            next_node = init_node
    return list(subgraph_nodes), list(subgraph_edges)


def draw_graph(sub_graph, edges, length, width):
    """
    To draw a grid graph.
    :param sub_graph:
    :param edges:
    :param length:
    :param width:
    :return:
    """
    import networkx as nx
    from pylab import rcParams
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import subplots_adjust
    subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    rcParams['figure.figsize'] = 14, 14

    G = nx.Graph()
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    pos = dict()
    index = 0
    for i in range(length):
        for j in range(width):
            G.add_node(index)
            pos[index] = (j, length - i)
            index += 1
    nx.draw_networkx_nodes(G, pos, node_size=100,
                           nodelist=range(33 * 33), node_color='gray')
    nx.draw_networkx_nodes(G, pos, node_size=100,
                           nodelist=sub_graph, node_color='b')
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=2)
    plt.axis('off')
    plt.show()


def minimal_spanning_tree(edges, weights, num_nodes):
    """
    Find the minimal spanning tree of a graph.
    :param edges: ndarray dim=(m,2) -- edges of the graph.
    :param weights: ndarray dim=(m,)  -- weights of the graph.
    :param num_nodes: int, number of nodes in the graph.
    :return: (the edge indices of the spanning tree)
    """
    try:
        from proj_module import mst
    except ImportError:
        print('cannot find this functions: proj_pcst')
        exit(0)
    select_indices = mst(edges, weights, num_nodes)
    return select_indices


def fast_pcst(edges, prizes, weights, root, g, pruning, epsilon, verbose):
    """
    Fast PCST algorithm using C11 language
    :param edges:
    :param prizes:
    :param root:
    :param weights:
    :param g:
    :param pruning:
    :param verbose:
    :param epsilon: to control the precision
    :return:
    """
    if not np.any(prizes):  # make sure
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    if not (weights > 0.).all():
        print('all weights must be positive.')
    # TODO to check variables.
    return c_proj_pcst(edges, prizes, weights, root, g, pruning, epsilon, verbose)


class HeadTailWrapper(object):
    """
    The Python wrapper for the head and tail approx. algorithms.
    """

    def __init__(self, edges, weights):
        """ head and tail approximation package
        :param edges: ndarray[mx2] edges of the input graph
        :param weights: weights of edges
        """
        self._edges = edges
        self._weights = weights
        if not (self._weights > 0.0).all():
            print('Error: all edge weights must be positive.')
            exit()

    def run_tail(self, x, g, s, budget, nu):
        """ Run tail approximation algorithm
        :param x: input vector for projection.
        :param g: number of connected components
        :param s: sparsity
        :param budget: budget
        :param nu: parameter nu used in the tail approx. algorithm.
        :return: (nodes, edges,proj_vector):
        projected nodes, edges and projected vector.
        """
        return tail_proj(self._edges, self._weights, x, g, s, budget, nu)

    def run_head(self, x, g, s, budget, delta):
        """ Run head approximation algorithm.
        :param x: input vector for projection
        :param g:  number of connected component
        :param s: sparsity parameter
        :param budget: budget
        :param delta: parameter delta used in the head approx. algorithm.
        :return: (nodes, edges,proj_vector):
        projected nodes, edges and projected vector.
        """
        return head_proj(self._edges, self._weights, x, g, s, budget, delta)


def head_proj(edges, weights, x, g, s, budget=None, delta=None, max_iter=None,
              err_tol=None, root=None, pruning=None, epsilon=None, verbose=None):
    """
    Head projection algorithm.
    :param edges: ndarray[mx2] edges of the input graph
    :param weights:  weights of edges
    :param x: input vector for projection
    :param g: number of connected component
    :param s: sparsity parameter
    :param budget:
    :param delta:
    :param max_iter: maximal iterations in head projection.
    :param err_tol: error tolerance for lower bound search bound.
    :param root: -1, no root for pcst
    :param pruning:
    :param epsilon:
    :param verbose:
    :return:
    """
    if budget is None:
        budget = 1. * (s - g)
    if delta is None:
        delta = 1. / 169.
    if max_iter is None:
        max_iter = 50
    if err_tol is None:
        err_tol = 1e-6
    if root is None:
        root = -1
    if pruning is None:
        pruning = 'strong'
    if verbose is None:
        verbose = 0
    if epsilon is None:
        epsilon = 1e-6
    # if it is a zero vector, then just return an empty graph
    if not np.any(x):
        p_x = np.zeros_like(x)  # projected vector
        return np.asarray([], dtype=int), np.asarray([], dtype=int), p_x
    # [re_nodes, re_edges, p_x]
    return c_proj_head(edges, weights, x, g, s, budget, delta, max_iter, err_tol, root, pruning, epsilon, verbose)


def tail_proj(edges, weights, x, g, s, budget=None, nu=None,
              max_iter=None, err_tol=None, root=None, pruning=None, verbose=None, epsilon=None):
    """
    Tail projection algorithm.
    :param edges: ndarray[mx2] edges of the input graph
    :param weights: weights of edges
    :param x: input vector for projection
    :param g: number of connected component
    :param s: sparsity parameter
    :param budget:
    :param nu:
    :param max_iter: maximal iterations
    :param err_tol:
    :param root: -1, no root for pcst
    :param pruning
    :param verbose
    :param epsilon
    :return:
    """
    if budget is None:
        budget = 1. * (s - g)
    if nu is None:
        nu = 2.5
    if max_iter is None:
        max_iter = 50
    if err_tol is None:
        err_tol = 1e-6
    if root is None:
        root = -1
    if pruning is None:
        pruning = 'strong'
    if verbose is None:
        verbose = 0
    if epsilon is None:
        epsilon = 1e-6
    # if it is a zero vector, then just return an empty graph
    if not np.any(x):
        p_x = np.zeros_like(x)  # projected vector
        return np.asarray([], dtype=int), np.asarray([], dtype=int), p_x
    # [re_nodes, re_edges, proj_x]
    return c_proj_tail(edges, weights, x, g, s, budget, nu, max_iter, err_tol, root, pruning, epsilon, verbose)
