"""
This module holds the code for simulating data on graph connectivity. It can then be used to learn a model upon which can be used for choosing the sampling rate in unionfind cluster editing. It could also be used for further research.
"""
#from python_algorithms.basic import union_find
#import importlib
#from union_find import *
import random
from math import log
import sys
import numpy as np
from numba import jit, njit
from numpy import random as rand


#######################################################
def initialize_union_find(n):
    uf_parent = np.arange(n, dtype=np.int64)
    uf_size = np.ones(n, dtype=np.int64)
    return np.asarray((uf_parent, uf_size))


#note: parent is the uf_parent part of the union find structure
@njit
def find(x, uf):
    parent = uf[0]

    while x != parent[x]:
        #path compression (halving):
        parent[x] = parent[parent[x]]
        x = parent[x]
    return(x)


@njit
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
#######################################################
@njit
def generate_edges(n):
    max_edges = np.int64((n * (n-1)) / 2)
    edges = np.zeros((max_edges,2), dtype=np.int64)
    k = 0
    for v1 in range(0,n):
        for v2 in range(0,v1):
            edges[k][0] = v1
            edges[k][1] = v2
            k = k + 1
    return edges


def sim_single_run_connected(n,p, edges):
    #uf_struct = union_find.UF(n-1)
    uf_struct = initialize_union_find(n)

    for e in rand.permutation(edges.shape[0]):
        if random.random() < p:
            v1 = edges[e][0]
            v2 = edges[e][1]
            #uf_struct.union(v1,v2)
            union(v1, v2, uf_struct)
            #this may end the simulation earlier than with union by rank :)
            if uf_struct[1][find(v1, uf_struct)] == n:
                return True
    return False
    # while uf_struct[1][v1] != n :
    #     vertex = vertex + 1
    #     if vertex == n-1:
    #         return True
   ##   return False


def sim_connectivity_rate(n,p,repetitions):
    edges = generate_edges(n)
    success_count = 0
    for i in range(1,repetitions):
        if sim_single_run_connected(n, p, edges):
            success_count = success_count + 1
    return success_count / repetitions

def sim_until_connected(n):
    uf_struct = initialize_union_find(n)
    edges = generate_edges(n)
    edge_count = int(0)
    # iterate over edge index (via edges.shape[0] array)
    for e in rand.permutation(edges.shape[0]):
        v1 = edges[e][0]
        v2 = edges[e][1]

        union(v1, v2, uf_struct)
        edge_count = edge_count + 1
        # if size of the parent of one of the nodes (now connected = same parent) is n,
        # graph is connected
        if uf_struct[1][find(v1, uf_struct)] == n:
            return edge_count

    return int(-1)

def sim_until_nearly_connected(n, p):
    uf_struct = initialize_union_find(n)
    edges = generate_edges(n)
    edge_count = int(0)
    # iterate over edge index (via edges.shape[0] array)
    for e in rand.permutation(edges.shape[0]):
        v1 = edges[e][0]
        v2 = edges[e][1]

        union(v1, v2, uf_struct)
        edge_count = edge_count + 1
        # if size of the parent of one of the nodes (now connected = same parent) is big enough
        # graph is connected to (at least) p-percent
        if (uf_struct[1][find(v1, uf_struct)] /n) >= p:
            return edge_count

def sim_until_nearly_connected_p(n, all_con_rates):
    uf_struct = initialize_union_find(n)
    edges = generate_edges(n)
    edge_count = int(0)
    c_i = 0
    # iterate over edge index (via edges.shape[0] array)
    for e in rand.permutation(edges.shape[0]):
        v1 = edges[e][0]
        v2 = edges[e][1]

        union(v1, v2, uf_struct)
        edge_count = edge_count + 1
        # if size of the parent of one of the nodes (now connected = same parent) is big enough
        # graph is connected to (at least) p-percent
        parent_size = uf_struct[1][find(v1, uf_struct)]
        while (parent_size/n) >= all_con_rates[c_i]:
            # print edge_count for all fitting connectivity rates
            print(edge_count, end=' ')
            c_i += 1
            # if parent_size/n is above maximum connectivity rate, end:
            if c_i == len(all_con_rates):
                print("")
                return


if __name__ == '__main__':
    # for lido experiments:
    arg_n = int(sys.argv[1])
    rand.seed(1234)
    for i in range(0,10000):
        sim_until_nearly_connected_p(arg_n, np.arange(0.05,0.95, 0.025))
