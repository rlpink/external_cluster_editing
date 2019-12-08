"""
This module implements several methods for calculating and outputting solutions of the unionfind_cluster_editing() algorithm.
It contains some methods to print solutions
and, more importantly, methods to merge solutions into one better solution.
There are 3 main algorithms: merge, repair and undo.
Two repair algorithms with differing complexity are implemented.
"""

from union_find import *
from math import log
import sys
import numpy as np
from numba import njit, jit
from numpy import random as rand
from model_sqrt import *
from numba.typed import Dict
import pandas as pd

def print_result(output_path, name, date):
    file = open(output_path + name, mode="a")
    file.write(str(date))
    file.close()

def print_zhk(output_path, merged, sizes):
    file = open(output_path + "zhk_sizes.txt", mode="a")
    for i in range(len(sizes)):
        if merged[i] == i:
            file.write(str(i) + " " + str(sizes[i]) + "\n")
    file.close()


def print_solution_costs(solution_costs, output_path):
    """
    This function outputs all sorted solution costs to a ifle named "..._solution_costs.txt".
    """
    sorted_costs = np.sort(solution_costs)
    print_to =  output_path + "solutions_v4.txt"
    with open(print_to, mode="a") as file:
        for cost in sorted_costs:
            file.write(str(cost))
            file.write("\n")


def all_solutions(solution_costs, parents, filename, missing_weight, n):
    """
    This function outputs all solutions, sorted by their costs, to a ifle named "all_solutions.txt".
    """
    cost_sorted_i = np.argsort(solution_costs)
    print_to = filename[:-4] + "_all_solutions_v4.txt"
    count = 1
    with open(print_to, mode="a") as file:
        file.write("filename: %s \nmissing_weight: %f \nn: %d\n" % (filename, missing_weight, n))
        for i in cost_sorted_i:
            file.write("%d. best solution with cost %f\n" % (count, solution_costs[i]))
            count += 1
            for j in range(0,n):
                file.write(f"{parents[i, j]} ")
            file.write("\n")



def merged_to_file(solutions, costs, filename, missing_weight, n, x, n_merges, output_path):
    """
    A function to write the merged solution(s) to a file, named like the input instance ending with _merged.txt.
    """
    print_to = output_path + "merged_v4.txt"
    with open(print_to, mode="a") as file:
        file.write("filename: %s \nmissing_weight: %f \nn: %d \nx (solutions merged): %d\nmerged solutions:\n" % (filename, missing_weight, n, x))
        for i in range(n):
            file.write(f"{solutions[0, i]} ")

def merged_short_print(solutions, costs, filename, missing_weight, n, x, n_merges):
    for j in range(n_merges):
        cluster_sizes = {}
        for i in range(n):
            curr = solutions[j, i]
            if curr in cluster_sizes:
                cluster_sizes[curr] += 1
            else:
                cluster_sizes[curr] = 1
        print(cluster_sizes)


#### merge: scan-variant ####


@njit
def weighted_decision_scan(x, y, connectivity, f_vertex_costs, f_sizes, f_parents):
    """
    This function is a helper function for merging functions. It generates a weight for cluster center x and another node y by counting the costs over all solutions for two scenarios:
    1: y is in the same cluster as x
    0: y is in another cluster
    The return value is between -1 and 1, -1 for certainly not connected, 1 for certainly connected. A value of 0 would indicate that connected or not connected would (in mean) yield the same costs (as in: the error is not big enough to make a difference).
    """
    sol_len = len(f_parents)
    sum_for_0 = 0
    sum_for_1 = 0
    count_0 = 0
    count_1 = 0
    for i in range(0,sol_len):
        x_cost = f_vertex_costs[i, x]
        y_cost = f_vertex_costs[i, y]
        if connectivity[i]:
            sum_for_1 += x_cost + y_cost
            count_1 += 1
        else:
            sum_for_0 += x_cost + y_cost
            count_0 += 1

    if count_0 > 0:
        cost_0 = sum_for_0/count_0
        if count_1 > 0:
            cost_1 = sum_for_1/count_1
            if cost_0 == 0 and cost_1 == 0:
                print("Warning: Both together and single get cost 0 - something went wrong!")
            else:
                return (cost_0 - cost_1) / (cost_0 + cost_1)

        else:
            # Falls kein Eintrag 1 gehört Knoten recht sicher nicht zum Cluster
            return -1.0
    else:
        # Falls kein Eintrag 0 gehört Knoten recht sicher zum Cluster
        return 1.0

    # Falls Rückgabe positiv: Entscheidung für 1 (zusammen), falls negativ: Entscheidung für 0 (getrennt).
    # Je näher Rückgabewert an 0, desto unsicherer die Entscheidung.
    # Falls kein voriger Fall eintritt (Häufigkeit entscheidet/ Verhältnis liegt vor):
    return 0.0

def merged_solution_scan(solution_costs, vertex_costs, parents, sizes, missing_weight, n, filename, output_path, union_threshold):
    """
    First merge algorithm. It calculates cluster masks for each cluster center:
    True, if the node is in the same component with cluster center,
    False otherwise.
    For these cluster masks, for each cluster center x and each other node y a weighted decision value is calculated. Is this weight better than the previous one, y gets assigned to new cluster center x. X then gets the weight of the maximum weight over all y, except if that is lower than its previous weight. Tree-like structures can emerge in such cases. Those trees are not handled yet, however they indicate a conflict in the solution, as a node that is both child and parent belongs to two distinct clusters.
    """
    sol_len = len(solution_costs)

    # Neue Lösung als Array anlegen:
    merged_sol = np.arange(n) #dtype = np.int64 not supported by numba
    merged_sizes = np.ones(n, dtype=np.int64)

    # Arrays anlegen für Vergleichbarkeit der Cluster:
    connectivity = np.zeros(sol_len, dtype=np.int8) #np.bool not supported
    graph_file = open(filename, mode="r")
    l = 0
    #wd_f = open(output_path + "wd_v4.txt", mode = "a")
    for line in graph_file:
        l += 1
        # Kommentar-Zeilen überspringen
        if line[0] == "#":
            continue
        splitted = line.split()
        nodes = np.array(splitted[:-1], dtype=np.int64)
        weight = np.float64(splitted[2])
        i = nodes[0]
        j = nodes[1]
        if weight < 0:
            continue
        # Fülle Cluster-Masken
        for x in range(sol_len):
            connectivity[x] = np.int8(parents[x, i] == parents[x, j])
        # Berechne Zugehörigkeit zu Cluster (bzw. oder Nicht-Zugehörigkeit)
        # Alle vorigen Knoten waren schon als Zentrum besucht und haben diesen Knoten daher schon mit sich verbunden (bzw. eben nicht) - Symmetrie der Kosten!
        wd = weighted_decision_scan(i, j, connectivity, vertex_costs, sizes, parents)
        #wd_f.write(str(l) + " " + str(wd) +"\n")
        # Falls Gewicht groß genug:
        if wd > union_threshold:
            union(i, j, merged_sol, merged_sizes)
    #wd_f.close()
    result = np.zeros((2,n))
    result[0] = merged_sol
    result[1] = merged_sizes

    return result


#### merge repair variants: with/without scan ####

@njit
def repair_merged_v4_nd(merged, merged_sizes, solution_costs, vertex_costs, parents, sizes, n, node_dgree, big_border):
    sol_len = len(solution_costs)
    ccs_mndgr = calculate_mean_nodedgr_nd(merged, merged_sizes, node_dgree)
    ccs = ccs_mndgr[0]
    mean_ndgree = ccs_mndgr[1]
    connectivity = np.zeros(sol_len, dtype=np.int8)

    for s_center_i in range(len(ccs)):
        # s_center soll klein genug sein
        s_center = ccs[s_center_i]
        if merged_sizes[s_center] > mean_ndgree[s_center_i] * big_border:
            continue
        # Detektiere und verbinde "Mini-Cluster" (Wurzel des Clusters soll verbunden werden);
        # Reparatur wird versucht, wenn die Größe des Clusters weniger als halb so groß ist wie der Knotengrad angibt, dh. die lokale Fehlerrate wäre bei über 50% in der Probleminstanz.
        best_fit = s_center
        min_mwc = 1.7976931348623157e+308
        for b_center_i in range(len(ccs)):
            # b_center soll groß genug sein
            b_center = ccs[b_center_i]
            if merged_sizes[b_center] <= mean_ndgree[b_center_i] * big_border:
                continue
            # Falls Cluster zusammen deutlich zu groß wären, überspringt diese Kombination direkt
            if merged_sizes[s_center] + merged_sizes[b_center] > 1.29 * mean_ndgree[b_center_i]:
                continue
            for x in range(0,sol_len):
                if parents[x, s_center] == parents[x, b_center]:
                    connectivity[x] = 1
                else:
                    connectivity[x] = 0
            # Berechne Gewicht:
            mwc = mean_weight_connected(s_center, connectivity, vertex_costs, sizes, parents)
            if mwc == -1:
                continue
            if mwc < min_mwc:
                # Aktualisieren von Minimalen Kosten
                min_mwc = mwc
                best_fit = b_center
        # Verbinde das Cluster mit dem Cluster, das im Mittel für s_center am günstigsten ist.
        union(s_center, best_fit, merged, merged_sizes)
    result = np.zeros((2,n), dtype=np.int64)
    result[0] = merged
    result[1] = merged_sizes
    return result


@njit
def mean_weight_connected(s_center, connectivity, vertex_costs, sizes, parents):
    sol_len = len(connectivity)
    mwc = 0.0
    count = 0
    for i in range(sol_len):
        if connectivity[i]:
            mwc += vertex_costs[i, s_center]
            count += 1
    if count == 0:
        return -1.0
    return mwc/count


@njit
def calculate_mean_nodedgr_array(merged, merged_sizes, node_dgree, cluster_centers):
    cluster_mean_nodedgr = np.zeros(len(cluster_centers), dtype=np.int64)
    for c in range(len(cluster_centers)):
        for i in range(len(merged)):
            if merged[i] == cluster_centers[c]:
                cluster_mean_nodedgr[c] += node_dgree[i]
        cluster_mean_nodedgr[c] /= merged_sizes[cluster_centers[c]]
    cmn_array = np.zeros(len(merged), dtype=np.int64)
    for i in range(len(cluster_centers)):
        c = cluster_centers[i]
        cmn_array[c] = cluster_mean_nodedgr[i]
    return cmn_array


@njit
def calculate_mean_nodedgr_nd(merged, merged_sizes, node_dgree):
    cluster_centers = pd.unique(merged)
    cluster_mean_nodedgr = np.zeros(len(cluster_centers), dtype=np.int64)
    for c in range(len(cluster_centers)):
        for i in range(len(merged)):
            if merged[i] == cluster_centers[c]:
                cluster_mean_nodedgr[c] += node_dgree[i]
        cluster_mean_nodedgr[c] /= merged_sizes[cluster_centers[c]]
    result = np.zeros((2,len(cluster_centers)), dtype=np.int64)
    result[0] = cluster_centers
    result[1] = cluster_mean_nodedgr
    return result


def repair_merged_v4_scan(merged, merged_sizes, solution_costs, vertex_costs, parents, sizes, n, node_dgree, big_border, filename):
    sol_len = len(solution_costs)
    cluster_centers = pd.unique(merged)
    mean_ndgree = calculate_mean_nodedgr_array(merged, merged_sizes, node_dgree, cluster_centers)
    connectivity = np.zeros(sol_len, dtype=np.int8)
    best_fits = np.zeros(n, dtype=np.int64)
    min_mwcs = np.zeros(n, dtype = np.float64)

    for i in range(n):
        best_fits[i] = -1
        min_mwcs[i] = 1.7976931348623157e+308

    graph_file = open(filename, mode="r")

    for line in graph_file:
        # Kommentar-Zeilen überspringen
        if line[0] == "#":
            continue
        splitted = line.split()
        nodes = np.array(splitted[:-1], dtype=np.int64)
        weight = np.float64(splitted[2])
        i = nodes[0]
        j = nodes[1]
        # Nur positive Kanten berücksichtigen
        if weight < 0:
            continue
        #Clusterzentren ermitteln
        s_center = merged[i]
        b_center = merged[j]
        # ggf. Benennung ändern (b: big, s: small)
        if merged_sizes[s_center] > merged_sizes[b_center]:
            tmp = s_center
            s_center = b_center
            b_center = tmp
        # Clustergrößen ermitteln
        s_center_s = merged_sizes[s_center]
        b_center_s = merged_sizes[b_center]

        if b_center_s < big_border * mean_ndgree[b_center]:
            continue
        if s_center_s >= big_border * mean_ndgree[s_center]:
            continue
        if s_center_s + b_center_s > 1.29 * mean_ndgree[s_center]:
            continue
        if s_center_s + b_center_s > 1.29 * mean_ndgree[b_center]:
            continue

        for x in range(0,sol_len):
            if parents[x, i] == parents[x, j]:
                connectivity[x] = 1
            else:
                connectivity[x] = 0
        # Berechne Gewicht:
        mwc = mean_weight_connected(s_center, connectivity, vertex_costs, sizes, parents)
        if mwc == -1:
            continue
        if mwc < min_mwcs[s_center]:
            # Aktualisieren von Minimalen Kosten
            min_mwcs[s_center] = mwc
            best_fits[s_center] = b_center
    # Laufe über alle großen Cluster (denen kleine zugewiesen wurden) und verbinde diese mit den günstigsten Kandidaten,
    # bis das Cluster (deutlich) zu voll wäre.
    bf_unique = pd.unique(best_fits)
    for b_center in bf_unique:
        # Wenn best_fits[i] == -1: wurde gar nicht befüllt (dh. i ist kein kleines Cluster oder wurde nie verbunden).
        if b_center == -1:
            continue
        sorted_candidates = priority_candidates(b_center, best_fits, min_mwcs)
        for s_center in sorted_candidates:
            # Check ob aktuelle Größe noch passt (im Unterschied zu oben: Dort wird nur geguckt ob die Größen -vor- dem ersten Union passen würden
            if merged_sizes[s_center] + merged_sizes[b_center] < 1.29 * mean_ndgree[b_center]:
                union(b_center, s_center, merged, merged_sizes)
    result = np.zeros((2,n), dtype=np.int64)
    result[0] = merged
    result[1] = merged_sizes
    return result


def priority_candidates(b_center, best_fits, min_mwcs):
    candidates = np.argwhere(best_fits == b_center).flatten()
    sorted_i = np.argsort(min_mwcs[candidates])
    return candidates[sorted_i]


def undo_merge_repair(merged, rep, merged_vc, rep_vc):
    new_sizes = np.zeros(len(merged), dtype=np.int64)
    for i in range(len(merged)):
        # Falls die Knotenkosten in der neuen Version echt größer sind als vorher, überschreibe den neuen Eintrag wieder mit dem alten Eintrag.
        if rep_vc[i] > merged_vc[i]:
            rep[i] = merged[i]
    for i in range(len(merged)):
        r = flattening_find(i, rep)
        new_sizes[r] += 1
    result = np.zeros((2,len(merged)), dtype=np.int64)
    result[0] = rep
    result[1] = new_sizes
    return result