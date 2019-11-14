import numpy as np
from numba import jit, njit

"""
This module contains all necessary functions to implement a union find (or disjoint-set) data structure.
"""

#note: parent is the uf_parent part of the union find structure
@njit
def find(x, parent):
    """
    This function gets a node x and the parent-part of a union find data structure.
    It returns the root of the tree that contains x. A tree compression technique called path halving is used.
    It is optimized with numba, using njit.
    """

    while x != parent[x]:
        #path compression (halving):
        parent[x] = parent[parent[x]]
        x = parent[x]
    return(x)


@njit
def union(x, y, parent, size):
    """
    This function is the heart of a union find data structure. It is used to union two nodes, x and y, given a uf-data structure consisting of parent (forest structure) and size (sizes of each components subtree). It uses find().
    """
    set_x = find(x, parent)
    #print("set_x ", set_x, " x ", x)
    set_y = find(y, parent)
    #print("set_y ", set_y, " y ", y)

    if set_x == set_y:
        #print("equal")
        return False

    if size[set_x] <= size[set_y]:
        parent[set_x] = set_y
        size[set_y] = size[set_y] + size[set_x]
        #print("set size")

    elif size[set_x] > size[set_y]:
        parent[set_y] = set_x
        size[set_x] = size[set_x] + size[set_y]
        #print("set size")
    return True

@njit
def rem_union(x, y, parent):
    """
    This function is a slightly faster version of the regular union.
    It does not need sizes and potentially shortens the search path in case x and y are already connected.
    """
    while parent[x] != parent[y]:
        if parent[x] < parent[y]:
            if x == parent[x]:
                parent[x] = parent[y]
                return
            z = parent[x]
            parent[x] = parent[y]
            x = z
        else:
            if y == parent[y]:
                parent[y] = parent[x]
                return
            z = parent[y]
            parent[y] = parent[x]
            y = z


#path compression for flattened uf-trees
@njit
def flattening_find(x, parent):
    """
    This function implements a find with (full) path compression, used to flatten a union-find tree if called on every leaf node.
    """
    if x != parent[x]:
        parent[x] = flattening_find(parent[x],parent)
    return parent[x]