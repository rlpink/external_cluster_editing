#from python_algorithms.basic import union_find
#import importlib
from union_find import *
import random
from math import log
import sys
import numpy
from numba import jit, njit
from numpy import random as rand

@njit
def generate_edges(n):
    max_edges = numpy.int64((n * (n-1)) / 2)
    edges = numpy.zeros((max_edges,2), dtype=numpy.int64)
    k = 0
    for v1 in range(0,n):
        for v2 in range(0,v1):
            edges[k][0] = v1
            edges[k][1] = v2
            k = k + 1
    return edges

# deletion_factor stellt ein, wie viele Kanten jeder Knoten maximal verliert
def disturb_cluster(n, edges, deletion_factor):
    rand_edges = rand.permutation(edges)
    n_edges = n-1
    vertexwise_del_edges = numpy.zeros(n, dtype=numpy.int64)
    i = 0

    file = open("graph_sim.txt", mode="a")

    for e in rand_edges:
        weight = 1
        # if both vertices can have one more edge deleted...
        if (((vertexwise_del_edges[e[0]] + 1) / n_edges) <= deletion_factor and
         ((vertexwise_del_edges[e[1]] + 1) / n_edges) <= deletion_factor):
            # set edge weight to -1
            weight = -1
            # count deleted edges for both vertices
            vertexwise_del_edges[e[0]] += 1
            vertexwise_del_edges[e[1]] += 1
            i += 1
        file.write("%d %d %d \n" % (e[0], e[1], weight))


def additional_edges(cbuckets, insertion_factor):
    file = open("graph_sim.txt", mode="")

    #todo: zusätzliche kanten auch direkt in datei schreiben. funktion sollte nicht zwei cliquen verbinden sondern über alle knoten des graphs in zufälliger reihenfolge laufen und analog zu disturb_cluster mit einem insertion_factor arbeiten.
