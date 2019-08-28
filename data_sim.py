#from python_algorithms.basic import union_find
#import importlib
from union_find import *
import random
from math import log
import sys
import numpy
from numba import jit
from numpy import random as rand


def generate_edges(n):
    max_edges = numpy.int64((n * (n-1)) / 2)
    edges = numpy.full((max_edges,2),(numpy.int64(0),numpy.int64(0)))
    k = 0
    for v1 in range(0,n):
        for v2 in range(0,v1):
            edges[k][0] = v1
            edges[k][1] = v2
            k = k + 1
    return edges

# deletion_factor stellt ein, wie viele Kanten jeder Knoten maximal verliert
def disturb_cluster(edges, deletion_factor):
    rand_edges = rand.permutation(edges)
    n = len(edges)
    vertexwise_del_edges = numpy.full(n, 0)
    edge_exists = numpy.full(n, True)
    i = 0

    for e in rand_edges:
        #todo: smarter abbruch sobald alle knoten "genug" kanten verloren haben
        # if both vertices can have one more edge deleted...
        if ((vertexwise_del_edges[e[0]] + 1) / (n-1)) <= deletion_factor
        and  (vertexwise_del_edges[e[1]] + 1) / (n-1)) <= deletion_factor):
            # mark edge as deleted
            edge_exists[i] = False
            # count deleted edges for both vertices
            vertexwise_del_edges[e[0]] += 1
            vertexwise_del_edges[e[1]] += 1
            i += 1

    #todo: statt return einfach in Datei schreiben! Für fehlende Kanten entspr. negative Gewichte vermerken
    return (rand_edges, edge_exists)


def additional_edges(clique_a, clique_b):
    #todo: zusätzliche kanten auch direkt in datei schreiben. funktion sollte nicht zwei cliquen verbinden sondern über alle knoten des graphs in zufälliger reihenfolge laufen und analog zu disturb_cluster mit einem insertion_factor arbeiten.
