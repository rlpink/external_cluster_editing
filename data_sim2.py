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
This module is intended to generate input data for transitivity clustering.
It constructs a file containing all edges that define an input-graph.
At its current version, only unweighted graphs can be generated.
An extension for weights is easily possible though, as primitive weights (1,-1) are already implemented.
"""

@njit
def generate_edges(n, offset):
    """
    This function generates edges for a fully connected (sub-)graph of size n.
    Parameter offset is intended to name the nodes to offset..n+offset (last excluded).
    """
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
@njit
def disturb_cluster(n, offset, edges, deletion_factor, optimal_costs):
    """
    This function deletes edges from an otherwise fully connected (sub-)graph by
    setting their weight to -1.
    Parameter deletion_factor controls how many edges are deleted:
    At most n * deletion_factor edges are deleted for every node.
    Parameter n is for the size of the (sub-)graph to be disturbed,
    parameter offset is needed to re-calculate indices from node names (other than 0..n),
    parameter edges takes the output of function generate_edges,
    parameter optimal_costs is to track the overall editing costs of the graph
    """
    rand_edges = rand.permutation(edges.shape[0])
    vertexwise_del_edges = np.zeros(n, dtype=np.int64)
    max_edges_out = deletion_factor * n
    i = 0

    for e in rand_edges:
        weight = 1
        # if both vertices can have one more edge deleted...
        if ((vertexwise_del_edges[edges[e][0]-offset] + 1) <= max_edges_out and
        (vertexwise_del_edges[edges[e][1]-offset] + 1) <= max_edges_out):
            # set edge weight to -1
            weight = -1
            optimal_costs[0] += 1
            # count deleted edges for both vertices
            vertexwise_del_edges[edges[e][0]-offset] += 1
            vertexwise_del_edges[edges[e][1]-offset] += 1
            i += 1
        print(edges[e][0], edges[e][1], weight)

@njit
def max_edges_in(i, cluster_bounds, insertion_factor):
    """
    This function calculates how many edges can be inserted for node i
    Parameter i is the name of the node
    parameter cluster_bounds gives the bounds of seperately constructed, once fully connected
    components of the graph,
    parameter insertion_factor determines how many edges can be inserted:
    For node i in cluster n_c, n_c * insertion_factor edges can be added.
    """
    # for j in range(1, len(cluster_bounds)):
    #     if(i < cluster_bounds[j] and i >= cluster_bounds[j-1]):
    #         break
    bounds = get_cluster_bounds(i, cluster_bounds)
    #n_c = cluster_bounds[j] - cluster_bounds[j-1]
    n_c = bounds[1] - bounds[0]
    return np.int64(n_c * insertion_factor)

@njit
def get_cluster_bounds(i, cluster_bounds):
    """
    This function calculates the cluster bounds for the cluster containing node i, given all cluster bounds.
    """
    con1 = np.where(i >= cluster_bounds)[0]
    j = con1[len(con1) -1]+1

    # for j in range(1, len(cluster_bounds)):
    #     if(i < cluster_bounds[j] and i >= cluster_bounds[j-1]):
    #         break
    return np.array([cluster_bounds[j-1], cluster_bounds[j]], dtype=np.int64)

@njit
def additional_edges(cluster_bounds, insertion_factor, optimal_costs):
    """
    This function adds edges to connect separately created clusters.
    It uses an insertion factor to controll the maximum number of edges that can be inserted for each node. For node i of a cluster with size n_i, at most n_i * insertion_factor edges can be inserted.
    Parameter optimal_cost is for keeping book of the editing costs.
    """
    n = cluster_bounds[len(cluster_bounds)-1]
    vertexwise_ins_edges = np.zeros(n, dtype=np.int64)
    vertexwise_max_edges = np.zeros(n, dtype=np.int64)

    for i in range(0,n):
        vertexwise_max_edges[i] = max_edges_in(i, cluster_bounds, insertion_factor)

    for v1 in rand.permutation(np.arange(0,n)):
        lower = get_cluster_bounds(v1, cluster_bounds)[0]
        upper = get_cluster_bounds(v1, cluster_bounds)[1]
        v2_arr = np.zeros((lower + (n-upper)), dtype= np.int64)
        k = 0
        for j in range(0, len(v2_arr)):
            if j < lower:
                v2_arr[j] = j
            else:
                v2_arr[j] = upper + k
                k += 1
        for v2_i in rand.permutation(v2_arr.shape[0]):
            v2 = v2_arr[v2_i]
            if (vertexwise_ins_edges[v1] + 1) > vertexwise_max_edges[v1]:
                break
            if (vertexwise_ins_edges[v2] + 1) <= vertexwise_max_edges[v2]:
                print(v1, v2, 1)
                vertexwise_ins_edges[v1] += 1
                vertexwise_ins_edges[v2] += 1
                optimal_costs[0] += 1

# cluster_sizes is formatted: np.array([0,<cluster_size1>,...,<cluster_sizek>])
def simulate_graph(seed, cluster_sizes, del_factor, ins_factor):
    """
    This function is the main function of this module. It generates fully connected clusters
    and disturbs them (deletes some edges, according to the del_factor).
    Afterwards, additional edges are generated, according to ins_factor.
    Besides the edges, it outputs a seed for the random generator, the deletion-
    and insertion-factor used, as well as (below all edges) the optimal editing costs for this graph, given that both del.- and ins.-factor are not too high.
    Parameter cluster_sizes takes an array with the size of each cluster.
    cluster_sizes[0] has to be set to 0, as it is used to calculate cluster boundaries.
    """
    rand.seed(seed)
    cluster_boundaries = np.cumsum(cluster_sizes)
    print("#seed:", seed)
    print("#deletion factor:", del_factor)
    print("#insertion factor:", ins_factor)
    optimal_costs = np.array([0])
    for c in range(0, len(cluster_sizes)-1):
        n_c = cluster_sizes[c+1]
        offset_c = cluster_boundaries[c]
        edges_c = generate_edges(n_c, offset_c)
        disturb_cluster(n_c, offset_c, edges_c, del_factor, optimal_costs)
    additional_edges(cluster_boundaries, ins_factor, optimal_costs)
    print("#optimal costs:", optimal_costs)

def generate_clusterarray(k_cluster, cluster_size):
    """
    Helper function to generate k_cluster clusters of size cluster_size which can be used as input for simulate_graph.
    """
    result = np.zeros(k_cluster + 1, dtype = np.int32)
    for i in range(1,len(result)):
        result[i] = cluster_size
    return result

#arg_n_c = int(sys.argv[1])
#arg_cluster_size = int(sys.argv[2])
#arg_del_fac = float(sys.argv[3])
#arg_ins_fac = float(sys.argv[4])
#clusters = generate_clusterarray(arg_n, arg_cluster_size)
#simulate_graph(123, clusters, arg_del_fac, arg_ins_fac)