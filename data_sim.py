#from python_algorithms.basic import union_find
#import importlib
from union_find import *
import random
from math import log
import sys
import numpy as np
from numba import jit, njit
from numpy import random as rand
from itertools import chain

"""
This module is an old version of module data_sim2.
It lacks some features of data_sim2, as it does not calculate the optimal costs.
Recommended not to use.
"""

@njit
def generate_edges(n, offset):
    max_edges = np.int64((n * (n-1)) / 2)
    edges = np.zeros((max_edges,2), dtype=np.int64)
    k = 0
    for v1 in range(0+offset,n+offset):
        for v2 in range(0+offset,v1):
            edges[k][0] = v1
            edges[k][1] = v2
            k = k + 1
    return edges

# deletion_factor stellt ein, wie viele Kanten jeder Knoten maximal verliert
def disturb_cluster(n, offset, edges, deletion_factor, file):
    rand_edges = rand.permutation(edges)
    vertexwise_del_edges = np.zeros(n, dtype=np.int64)
    max_edges_out = deletion_factor * n
    i = 0

    for e in rand_edges:
        weight = 1
        # if both vertices can have one more edge deleted...
        if ((vertexwise_del_edges[e[0]-offset] + 1) <= max_edges_out and
        (vertexwise_del_edges[e[1]-offset] + 1) <= max_edges_out):
            # set edge weight to -1
            weight = -1
            # count deleted edges for both vertices
            vertexwise_del_edges[e[0]-offset] += 1
            vertexwise_del_edges[e[1]-offset] += 1
            i += 1
        file.write("%d %d %d \n" % (e[0], e[1], weight))

@njit
def max_edges_in(i, cluster_bounds, insertion_factor):
    for j in range(1, len(cluster_bounds)):
        if(i < cluster_bounds[j] and i >= cluster_bounds[j-1]):
            break
    n_c = cluster_bounds[j] - cluster_bounds[j-1]
    return np.int64(n_c * insertion_factor)

@njit
def get_cluster_bounds(i, cluster_bounds):
    #todo: speed-up durch binary search
    for j in range(1, len(cluster_bounds)):
        if(i < cluster_bounds[j] and i >= cluster_bounds[j-1]):
            break
    return np.array([cluster_bounds[j-1], cluster_bounds[j]], dtype=np.int64)

def additional_edges(cluster_bounds, insertion_factor, file):
    n = cluster_bounds[len(cluster_bounds)-1]
    vertexwise_ins_edges = np.zeros(n, dtype=np.int64)
    vertexwise_max_edges = np.zeros(n, dtype=np.int64)

    for i in range(0,n):
        vertexwise_max_edges[i] = max_edges_in(i, cluster_bounds, insertion_factor)

    for v1 in rand.permutation(range(0,n)):
        lower = get_cluster_bounds(v1, cluster_bounds)[0]
        upper = get_cluster_bounds(v1, cluster_bounds)[1]
        for v2 in rand.permutation(np.fromiter(chain(range(0,lower), range(upper, n)), dtype= np.int64)):
            if (vertexwise_ins_edges[v1] + 1) > vertexwise_max_edges[v1]:
                break
            if (vertexwise_ins_edges[v2] + 1) <= vertexwise_max_edges[v2]:
                file.write("%d %d %d \n" % (v1, v2, 1))
                vertexwise_ins_edges[v1] += 1
                vertexwise_ins_edges[v2] += 1

# cluster_sizes is formatted: np.array([0,<cluster_size1>,...,<cluster_sizek>])
def simulate_graph(seed, cluster_sizes, del_factor, ins_factor):
    rand.seed(seed)
    cluster_boundaries = np.cumsum(cluster_sizes)
    with open("graph_sim.txt", mode="a") as file:
        file.write("#seed: %d\n" % seed)
        file.write("#del_factor: %f\n" % del_factor)
        file.write("#ins_factor: %f\n" % ins_factor)
        for c in range(0, len(cluster_sizes)-1):
            n_c = cluster_sizes[c+1]
            offset_c = cluster_boundaries[c]
            edges_c = generate_edges(n_c, offset_c)
            disturb_cluster(n_c, offset_c, edges_c, del_factor, file)
        additional_edges(cluster_boundaries, ins_factor, file)

simulate_graph(123, np.array([0,30,30,30]), 2/9, 2/9)