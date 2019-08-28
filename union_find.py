import numpy

def initialize_union_find(n):
    uf_parent = numpy.arange(n, dtype=numpy.int64)
    uf_size = numpy.ones(n, dtype=numpy.int64)
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
    set_x = find(x, uf)
    #print("set_x ", set_x, " x ", x)
    set_y = find(y, uf)
    #print("set_y ", set_y, " y ", y)
    parent = uf[0]
    size = uf[1]

    if set_x == set_y:
        #print("equal")
        return

    if size[set_x] <= size[set_y]:
        parent[set_x] = set_y
        size[set_y] = size[set_y] + size[set_x]
        #print("set size")

    elif size[set_x] > size[set_y]:
        parent[set_y] = set_x
        size[set_x] = size[set_x] + size[set_y]
        #print("set size")