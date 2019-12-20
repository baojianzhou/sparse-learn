# -*- coding: utf-8 -*-
import numpy as np
from sparse_learning.proj_algo import head_proj
from sparse_learning.proj_algo import tail_proj
from sparse_learning.fast_pcst import fast_pcst
from sparse_learning.graph_utils import simu_graph

def test_proj_head():
    print('-' * 100)
    edges, weights = simu_graph(25)  # get grid graph
    sub_graph = [6, 7, 8, 9]
    x = np.random.normal(0.0, 0.1, 25)
    x[sub_graph] = 5.
    n, m = len(weights), edges.shape[1]
    re = head_proj(edges=edges, weights=weights, x=x, g=1, s=4, budget=3.,
                   delta=1. / 169., err_tol=1e-6, max_iter=30, root=-1,
                   pruning='strong', epsilon=1e-6, verbose=0)
    re_nodes, re_edges, p_x = re
    print('test1 result head nodes: ', re_nodes)
    print('test1 result head edges: ', re_edges)
    print(p_x)
    re = head_proj(edges=edges, weights=weights, x=np.zeros(n), g=1, s=4,
                   budget=3., delta=1. / 169., err_tol=1e-6, max_iter=30,
                   root=-1, pruning='strong', epsilon=1e-6, verbose=0)
    re_nodes, re_edges, p_x = re
    print('test2 result head nodes: ', re_nodes)
    print('test2 result head edges: ', re_edges)
    print(p_x)
