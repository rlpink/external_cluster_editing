import numpy as np
from numba import jit, njit

def initialize_union_find(n):
    uf_parent = np.arange(n, dtype=np.int64)
    uf_size = np.ones(n, dtype=np.int64)
    return np.asarray((uf_parent, uf_size))


#note: parent is the uf_parent part of the union find structure
@njit
def find(x, parent):

    while x != parent[x]:
        #path compression (halving):
        parent[x] = parent[parent[x]]
        x = parent[x]
    return(x)


#@njit
def union(x, y, parent, size):
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

#path compression for flattened uf-trees
def flattening_find(x, parent):
    if x != parent[x]:
        parent[x] = flattening_find(parent[x],parent)
    return parent[x]