#from python_algorithms.basic import union_find
from union_find import *
import random
from math import log
import sys
import numpy
from numpy import random as rand

def generate_edges(n):
    max_edges = (n * (n-1)) / 2
    edges = numpy.full((max_edges, 2), (0,0))
    k = 0
    for v1 in range(0,n-1):
        for v2 in range(0,n-1):
            if k == max_edges return edges
            edges[k][0] = v1
            edges[k][1] = v2
            k = k + 1


def sim_single_run_connected(n,p, edges):
    #uf_struct = union_find.UF(n-1)
    uf_struct = initialize_union_find(n)

    for e in rand.permutation(edges.shape[0]):
        if random.random() < p:
            edges[e][0] = v1
            edges[e][1] = v2
            #uf_struct.union(v1,v2)
            union(v1, v2, uf_struct)
            #this may end the simulation earlier than with union by rank :)
            if uf_struct[1][v1] == n or uf_struct[1][v2] == n
                return True
    return False
    # while uf_struct[1][v1] != n :
    #     vertex = vertex + 1
    #     if vertex == n-1:
    #         return True
   ##   return False


def asymptotic_p(n):
    return (log(n)+4)/(n-1) # 4 = 2*c, c = 2 for P=0.9997


def sim_connectivity_rate(n,p,repetitions):
    edges = generate_edges(n)
    success_count = 0
    for i in range(1,repetitions):
        if sim_single_run_connected(n, p, edges):
            success_count = success_count + 1
    return success_count / repetitions

def sim_until_connected(n, repetitions):
    uf_struct = initialize_union_find(n)
    edges = generate_edges(n)
    edge_count = 0
    for e in rand.permutation(edges.shape[0]):
        edges[e][0] = v1
        edges[e][1] = v2

        union(v1, v2, uf_struct)
        edge_count = edge_count + 1
        # if size of v1 or v2 == n: graph is connected
        if uf_struct[1][v1] == n or uf_struct[1][v2] == n
            return edge_count


#todo: def new routine for calculation of "perfect" p = p* for each n
#todo: def new routine for calculation of n where asymptotic p ~ p*
#todo: adjust part for lido scripts
#arg_n = int(sys.argv[1])
#print(calculate_p_for_con_rate_above_z(arg_n))

