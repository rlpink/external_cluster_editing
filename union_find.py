import numpy

def initialize_union_find(n):
    uf_parent = numpy.arange(n, dtype=numpy.int64)
    uf_size = numpy.ones(dtype=numpy.int64)
    return (uf_parent, uf_size)

#note: parent is the uf_parent part of the union find structure
def find(x, uf):
    parent = uf[0]
    while x != parent[x]:
        #path compression (halving):
        parent[x] = parent[parent[x]]
        x = parent[x]
    return(x)

def union(x, y, uf):
    parent = uf[0]
    size = uf[1]
    set_x = find(x, parent)
    set_y = find(y, parent)

    if set_x == set_y:
        return

    if size[set_x] < size[set_y]:
        parent[set_x] = set_y
        size[set_y] = size[set_y] + size[set_x]

    elif size[set_x] > size[set_y]:
        parent[set_y] = set_x
        size[set_x] = size[set_x] + size[set_y]