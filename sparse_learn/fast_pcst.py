# -*- coding: utf-8 -*-
__all__ = ['fast_pcst']
import numpy as np

try:
    from c_sparse_learn import c_proj_pcst
except ImportError:
    print('cannot find this function: proj_pcst')
    exit(0)


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
