"""
This module implements several methods for calculating and outputting solutions of the unionfind_cluster_editing() algorithm.
It contains two methods for the (best) generated raw solutions,
and, more importantly, methods to merge solutions into one better solution.
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

def best_solution(solution_costs, parents, filename, missing_weight, n, x):
    """
    This function outputs the best generated solution to a file named "result.txt".
    """
    costs = solution_costs.min()
    best = parents[solution_costs.argmin()]
    file = open("result.txt", mode="a")
    with file:
        file.write("filename: %s \nmissing_weight: %f \nn: %d \nx (solutions generated): %d\nbest solution found:\n" % (filename, missing_weight, n, x))
        file.write(f"costs: {costs}\n")
        for i in range(0,n):
            file.write(f"{best[i]} ")

def print_solution_costs(solution_costs, filename):
    """
    This function outputs all sorted solution costs to a ifle named "..._solution_costs.txt".
    """
    sorted_costs = np.sort(solution_costs)
    print_to = filename[:-4] + "_solution_costs_v5.txt"
    with open(print_to, mode="a") as file:
        for cost in sorted_costs:
            file.write(str(cost))
            file.write("\n")

def all_solutions(solution_costs, parents, filename, missing_weight, n):
    """
    This function outputs all solutions, sorted by their costs, to a ifle named "all_solutions.txt".
    """
    cost_sorted_i = np.argsort(solution_costs)
    print_to = filename[:-4] + "_all_solutions_v5.txt"
    count = 1
    with open(print_to, mode="a") as file:
        file.write("filename: %s \nmissing_weight: %f \nn: %d\n" % (filename, missing_weight, n))
        for i in cost_sorted_i:
            file.write("%d. best solution with cost %f\n" % (count, solution_costs[i]))
            count += 1
            for j in range(0,n):
                file.write(f"{parents[i, j]} ")
            file.write("\n")

@njit
def weighted_decision(x, y, cluster_masks, f_vertex_costs, f_sizes, f_parents):
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
        if cluster_masks[i, y] == 0:
            sum_for_0 += x_cost + y_cost
            count_0 += 1
        else:
            sum_for_1 += x_cost + y_cost
            count_1 += 1

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


@njit
def merged_solution(solution_costs, vertex_costs, parents, sizes, missing_weight, n):
    """
    First merge algorithm. It calculates cluster masks for each cluster center:
    True, if the node is in the same component with cluster center,
    False otherwise.
    For these cluster masks, for each cluster center x and each other node y a weighted decision value is calculated. Is this weight better than the previous one, y gets assigned to new cluster center x. X then gets the weight of the maximum weight over all y, except if that is lower than its previous weight. Tree-like structures can emerge in such cases. Those trees are not handled yet, however they indicate a conflict in the solution, as a node that is both child and parent belongs to two distinct clusters.
    """
    sol_len = len(solution_costs)

    # Neue Lösung als Array anlegen:
    merged_sol = np.arange(n) #dtype = np.int64 not supported by numba

    # Arrays anlegen für Vergleichbarkeit der Cluster:
    cluster_masks = np.zeros((sol_len,n), dtype=np.int8) #np.bool not supported

    for j in range(n):
        # Fülle Cluster-Masken
        for i in range(sol_len):
            # Jede Cluster-Maske enthält "True" überall, wo parents
            # denselben Wert hat wie an Stelle j, sonst "False"
            for k in range(n):
                cluster_masks[i, k] = np.int8(parents[i, k] == parents[i, j])

        # Berechne Zugehörigkeit zu Cluster (bzw. oder Nicht-Zugehörigkeit)
        # Alle vorigen Knoten waren schon als Zentrum besucht und haben diesen Knoten daher schon mit sich verbunden (bzw. eben nicht) - Symmetrie der Kosten!
        for k in range(j+1,n):
            # Cluster-Zentrum wird übersprungen (dh. verweist möglicherweise noch auf anderes Cluster!)
            if k == j:
                continue
            wd = weighted_decision(j, k, cluster_masks, vertex_costs, sizes, parents)
            # Falls Gewicht groß genug:
            if wd > 0.05:
                rem_union(j, k, merged_sol)

    return merged_sol


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


def merged_solution_scan(solution_costs, vertex_costs, parents, sizes, missing_weight, n, filename):
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

    for line in graph_file:
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
        # Falls Gewicht groß genug:
        if wd > 0.05:
            rem_union(i, j, merged_sol)

    return merged_sol


@njit
def repair_merged(merged, merged_sizes, solution_costs, vertex_costs, parents, sizes, n, node_dgree):
    sol_len = len(solution_costs)
    # Arrays anlegen für Vergleichbarkeit der Cluster:
    cluster_masks = np.zeros((sol_len,n), dtype=np.int8) #np.bool not supported

    for i in range(n):
        # Detektiere und verbinde "Mini-Cluster" (Wurzel des Clusters soll verbunden werden);
        # Reparatur wird versucht, wenn die Größe des Clusters weniger als halb so groß ist wie der Knotengrad angibt, dh. die lokale Fehlerrate wäre bei über 50% in der Probleminstanz.
        if merged[i] == i and merged_sizes[i] < 0.5*node_dgree[i]:
            max_wd = -1
            best_fit = i
            # Fülle Cluster-Masken
            for x in range(0,sol_len):
                for j in range(n):
                    # Jede Cluster-Maske enthält "True" überall, wo parents
                    # denselben Wert hat wie an Stelle j, sonst "False"
                    cluster_masks[x, j] = np.int8(parents[x, i] == parents[x, j])

            for j in range(n):
                # Überspringe bereits verbundene Knoten und sich selbst
                if merged[i] == merged[j]:
                    continue
                # Berechne Gewicht:
                wd = weighted_decision(i, j, cluster_masks, vertex_costs, sizes, parents)
                # Aktualisiere ggf. best-passenden Knoten
                if wd > max_wd:
                    max_wd = wd
                    best_fit = j
            # ggf. Modifikation, nur union falls auch max_wd passt.
            #if max_wd > 0.1:
            union(i, best_fit, merged, merged_sizes)
    result = np.zeros((2,n), dtype=np.int64)
    result[0] = merged
    result[1] = merged_sizes

    return result

def get_cluster_centers_big(merged, merged_sizes, node_dgree, split):
    big_ccs = {}
    for i in range(len(merged)):
        if merged_sizes[merged[i]] >= node_dgree[merged[i]] * split:
            big_ccs[merged[i]] = merged_sizes[merged[i]]
    return big_ccs


def get_cluster_centers_small(merged, merged_sizes, node_dgree, split):
    small_ccs = {}
    for i in range(len(merged)):
        if merged_sizes[merged[i]] < node_dgree[merged[i]] * split:
            small_ccs[merged[i]] = merged_sizes[merged[i]]
    return small_ccs

def get_second_center(merged, big_ccs):
    second_cc = {}
    for center in big_ccs.keys():
        # Durchlaufe solange andere Knoten bis einer aus dem selben Cluster gefunden wurde
        for i in range(len(merged)):
            # nicht der selbe Knoten ist gesucht
            if i == center:
                continue
            # sondern der erste, der einen anderen Index hat aber den selben Eintrag:
            if merged[i] == merged[center]:
                second_cc[center] = i
                break
    return second_cc

@njit
def weighted_decision_2(s_center, b_center, sb_center, connectivity, vertex_costs, sizes, parents):
    costs_0 = 0.0
    costs_1 = 0.0
    count_0 = 0
    count_1 = 0
    for x in range(0, len(connectivity)):
        if connectivity[x] == -1:
            costs_1 += 0.5 * vertex_costs[x, s_center] + vertex_costs[x, b_center] + vertex_costs[x, b_center]
        elif connectivity[x] == -2:
            costs_1 += 0.5 * vertex_costs[x, s_center] + vertex_costs[x, sb_center] + vertex_costs[x, sb_center]
        elif connectivity[x] == 1:
            costs_1 += vertex_costs[x, s_center] + vertex_costs[x, b_center] + vertex_costs[x, sb_center]
            count_1 += 1
        else:
            costs_0 += vertex_costs[x, s_center] + vertex_costs[x, b_center] + vertex_costs[x, sb_center]
            count_0 += 1

    if count_0 > 0:
        cost_0 = costs_0/count_0
        if count_1 > 0:
            cost_1 = costs_1/count_1
            if cost_0 == 0 and cost_1 == 0:
                print("Warning: Both together and single get cost 0 - something went wrong!")
            else:
                return (cost_0 - cost_1) / (cost_0 + cost_1)

        else:
            # Falls kein Eintrag 1, gehört Knoten recht sicher nicht zum Cluster
            return -1.0
    else:
        # Falls kein Eintrag 0, gehört Knoten recht sicher zum Cluster
        return 1.0


def repair_merged_v2(merged, merged_sizes, solution_costs, vertex_costs, parents, sizes, n, node_dgree):
    sol_len = len(solution_costs)
    # Arrays anlegen für Vergleichbarkeit der Cluster:
    connectivity = np.zeros(sol_len, dtype=np.int8) #np.bool not supported
    big_ccs = get_cluster_centers_big(merged, merged_sizes, node_dgree, 0.3)
    small_ccs = get_cluster_centers_small(merged, merged_sizes, node_dgree, 0.3)
    second_big_cc = get_second_center(merged, big_ccs)

    for s_center in small_ccs.keys():
        # Detektiere und verbinde "Mini-Cluster" (Wurzel des Clusters soll verbunden werden);
        # Reparatur wird versucht, wenn die Größe des Clusters weniger als halb so groß ist wie der Knotengrad angibt, dh. die lokale Fehlerrate wäre bei über 50% in der Probleminstanz.
        max_wd = -1
        best_fit = s_center
        # Fülle connectivity-Array (0: keine Verbindung zu Cluster; 1: eine Verbindung, 2: zwei Verbindungen)
        for b_center in big_ccs.keys():
            # Falls Cluster zusammen deutlich zu groß wären, überspringe diese Kombination direkt
            if merged_sizes[s_center] + merged_sizes[b_center] > 1.5 * node_dgree[b_center]:
                continue
            for x in range(0,sol_len):
                if parents[x, b_center] != parents[x, second_big_cc[b_center]]:
                    connectivity[x] = -1
                    continue
                if parents[x, s_center] == parents[x, b_center]:
                    connectivity[x] = 1
                else:
                    connectivity[x] = 0
            # Berechne Gewicht:
            wd = weighted_decision_2(s_center, b_center, second_big_cc[b_center], connectivity, vertex_costs, sizes, parents)
            # Aktualisiere ggf. best-passenden Knoten
            if wd > max_wd:
                max_wd = wd
                best_fit = b_center
        # ggf. Modifikation, nur union falls auch max_wd passt.
        if max_wd > 0.05:
            union(s_center, best_fit, merged, merged_sizes)
    result = np.zeros((2,n), dtype=np.int64)
    result[0] = merged
    result[1] = merged_sizes
    return result

def repair_merged_v3(merged, merged_sizes, solution_costs, vertex_costs, parents, sizes, n, node_dgree):
    sol_len = len(solution_costs)
    ccs = calculate_mean_nodedgr(merged, merged_sizes, node_dgree)
    second_big_cc = get_second_center(merged, ccs)
    connectivity = np.zeros(sol_len, dtype=np.int8)

    for s_center in ccs.keys():
        # s_center soll klein genug sein
        if merged_sizes[s_center] > ccs[s_center] * 0.35:
            continue
        # Detektiere und verbinde "Mini-Cluster" (Wurzel des Clusters soll verbunden werden);
        # Reparatur wird versucht, wenn die Größe des Clusters weniger als halb so groß ist wie der Knotengrad angibt, dh. die lokale Fehlerrate wäre bei über 50% in der Probleminstanz.
        best_fit = s_center
        max_wd = -0.05
        for b_center in ccs.keys():
            # b_center soll groß genug sein
            if merged_sizes[b_center] <= ccs[b_center] * 0.35:
                continue
            # Falls Cluster zusammen deutlich zu groß wären, überspringe diese Kombination direkt
            if merged_sizes[s_center] + merged_sizes[b_center] > 1.5 * ccs[b_center]:
                continue
            for x in range(0,sol_len):
                if parents[x, b_center] != parents[x, second_big_cc[b_center]]:
                    connectivity[x] = -1
                    continue
                if parents[x, s_center] == parents[x, b_center]:
                    connectivity[x] = 1
                else:
                    connectivity[x] = 0
            # Berechne Gewicht:
            wd = weighted_decision_2(s_center, b_center, second_big_cc[b_center], connectivity, vertex_costs, sizes, parents)
            # Aktualisiere ggf. best-passenden Knoten
            if wd > max_wd:
                max_wd = wd
                best_fit = b_center
        # Verbinde das Cluster mit dem Cluster, das lokal betrachtet die geringsten Knotenkosten einbrachte.
        union(s_center, best_fit, merged, merged_sizes)
    result = np.zeros((2,n), dtype=np.int64)
    result[0] = merged
    result[1] = merged_sizes
    return result

@njit
def repair_merged_v3_nd(merged, merged_sizes, solution_costs, vertex_costs, parents, sizes, n, node_dgree):
    sol_len = len(solution_costs)
    ccs_mndgr = calculate_mean_nodedgr_nd(merged, merged_sizes, node_dgree)
    ccs = ccs_mndgr[0]
    mean_ndgree = ccs_mndgr[1]
    second_big_cc = get_second_center_nd(merged, ccs)
    connectivity = np.zeros(sol_len, dtype=np.int8)

    for s_center_i in range(len(ccs)):
        # s_center soll klein genug sein
        s_center = ccs[s_center_i]
        if merged_sizes[s_center] > mean_ndgree[s_center_i] * 0.35:
            continue
        # Detektiere und verbinde "Mini-Cluster" (Wurzel des Clusters soll verbunden werden);
        # Reparatur wird versucht, wenn die Größe des Clusters weniger als halb so groß ist wie der Knotengrad angibt, dh. die lokale Fehlerrate wäre bei über 50% in der Probleminstanz.
        best_fit = s_center
        max_wd = 0
        for b_center_i in range(len(ccs)):
            # b_center soll groß genug sein
            b_center = ccs[b_center_i]
            if merged_sizes[b_center] <= mean_ndgree[b_center_i] * 0.35:
                continue
            # Falls Cluster zusammen deutlich zu groß wären, überspringt diese Kombination direkt
            if merged_sizes[s_center] + merged_sizes[b_center] > 1.5 * mean_ndgree[b_center_i]:
                continue
            for x in range(0,sol_len):
                # Unterscheide vier Fälle: -1/-2: s_center nur mit einem verbunden; 1: mit beiden; 0: mit keinem

                if parents[x, b_center] != parents[x, second_big_cc[b_center_i]]:
                    if parents[x, s_center] == parents[x, b_center]:
                        connectivity[x] = -1
                    elif parents[x, s_center] == parents[x, second_big_cc[b_center_i]]:
                        connectivity[x] = -2
                    continue
                if parents[x, s_center] == parents[x, b_center]:
                    connectivity[x] = 1
                else:
                    connectivity[x] = 0
            # Berechne Gewicht:
            wd = weighted_decision_2(s_center, b_center, second_big_cc[b_center_i], connectivity, vertex_costs, sizes, parents)
            # Aktualisiere ggf. best-passenden Knoten
            if wd > max_wd:
                max_wd = wd
                best_fit = b_center
        # Verbinde das Cluster mit dem Cluster, das lokal betrachtet die geringsten Knotenkosten einbrachte.
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
def mean_weight_connected2(s_center, b_center, connectivity, vertex_costs, sizes, parents):
    sol_len = len(connectivity)
    mwc = 0.0
    mwd = 0.0
    count = 0
    countd = 0
    for i in range(sol_len):
        if connectivity[i]:
            mwc += vertex_costs[i, s_center] + vertex_costs[i, b_center]
            count += 1
        else:
            mwd += vertex_costs[i, s_center] + vertex_costs[i, b_center]
            countd += 1
    if count == 0:
        return -1.0
    elif countd == 0:
        return 1
    cost_1 = mwc/count
    cost_0 = mwd/countd
    return (cost_0 - cost_1) / (cost_0 + cost_1)

@njit
def repair_merged_v4_nd_rem(merged, merged_sizes, solution_costs, vertex_costs, parents, sizes, n, node_dgree, big_border):
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
        # Detektiere und verbinde "Mini-Cluster" (Wurzel des Clusters soll verbunden werden).
        best_fit = s_center
        min_mwc = 1.7976931348623157e+308
        for b_center_i in range(len(ccs)):
            # b_center soll groß genug sein
            b_center = ccs[b_center_i]
            if merged_sizes[b_center] <= mean_ndgree[b_center_i] * big_border:
                continue
            # Falls Cluster zusammen deutlich zu groß wären, überspringt diese Kombination direkt.
            # zu groß: mehr als 0.29 zusätzlich
            # wegen 2/9 Fehlerrate maximal die von den 7/9 übrigen Kanten jeweils fehlen darf.
            if merged_sizes[s_center] + merged_sizes[b_center] > 1.29 * mean_ndgree[b_center_i]:
                continue
            for x in range(0,sol_len):
                if parents[x, s_center] == parents[x, b_center]:
                    connectivity[x] = 1
                else:
                    connectivity[x] = 0
            # Berechne Gewicht:
            mwc = mean_weight_connected(s_center, connectivity, vertex_costs, sizes, parents)
            # Aktualisiere ggf. best-passenden Knoten und minimalen mwc
            if mwc == -1:
                continue
            if mwc < min_mwc:
                min_mwc = mwc
                best_fit = b_center

        # Verbinde das Cluster mit dem Cluster, das im Mittel für s_center am günstigsten ist.
        rem_union(s_center, best_fit, merged)
        # Wg. Rem: aktualisiere Größe direkt in Repräsentanten von später erneut betrachtetem best_fit
        merged_sizes[best_fit] += merged_sizes[s_center]
    return merged



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


def repair_merged_v4_rem_scan(merged, merged_sizes, solution_costs, vertex_costs, parents, sizes, n, node_dgree, big_border, filename):
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
                rem_union(b_center, s_center, merged)
                merged_sizes[b_center] += merged_sizes[s_center]
    return merged


def priority_candidates(b_center, best_fits, min_mwcs):
    candidates = np.argwhere(best_fits == b_center).flatten()
    sorted_i = np.argsort(min_mwcs[candidates])
    return candidates[sorted_i]


@njit
def repair_merged_wd(merged, merged_sizes, solution_costs, vertex_costs, parents, sizes, n, node_dgree, big_border):
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
        max_wd = -1
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
            wd = mean_weight_connected2(s_center, b_center, connectivity, vertex_costs, sizes, parents)
            if wd > max_wd:
                # Aktualisieren von Minimalen Kosten
                max_wd = wd
                best_fit = b_center

        # if best_fit == s_center:
        #     print("Knoten %d wurde nicht verbunden.\n" % (s_center))
        # Verbinde das Cluster mit dem Cluster, das im Mittel für s_center am günstigsten ist.
        rem_union(s_center, best_fit, merged)
        # Wg. Rem: aktualisiere Größe direkt in Repräsentanten von später erneut betrachtetem best_fit
        merged_sizes[best_fit] += merged_sizes[s_center]
    return merged


@njit
def check_if_flat(solution):
    for i in range(len(solution)):
        # Prüfe, ob Knoten i Wurzel ist oder Kind 1. Ebene
        if solution[i] != i and solution[solution[i]] != solution[i]:
            # falls es beides nicht ist, ist der Baum nicht flach!
            return False
    return True
@njit
def mean_node_weight(node, connectivity, vertex_costs, solutions, sizes, n):
    mnw_con = 0.0
    count = np.sum(connectivity)
    if count == 0:
        return -1
    for x in range(len(connectivity)):
        if not connectivity[x]:
            continue
        root = solutions[x, node]
        cluster_costs = 0.0
        for i in range(n):
            if solutions[x, i] == root:
                cluster_costs += vertex_costs[x, i]
        if sizes[x, root] != 0:
            mnw = cluster_costs / sizes[x, root]
        else:
            print("Sizes = 0 bei Lösung:", x)
            print("An Stelle:", root)
        mnw_con += mnw
    mnw_con = mnw_con / count
    return mnw_con

@njit
def repair_merged_v6_nd_rem(merged, merged_sizes, solution_costs, vertex_costs, parents, sizes, n, node_dgree, big_border):
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
        # Detektiere und verbinde "Mini-Cluster" (Wurzel des Clusters soll verbunden werden).
        best_fit = s_center
        min_mwc = 1.7976931348623157e+308
        for b_center_i in range(len(ccs)):
            # b_center soll groß genug sein
            b_center = ccs[b_center_i]
            if merged_sizes[b_center] <= mean_ndgree[b_center_i] * big_border:
                continue
            # Falls Cluster zusammen deutlich zu groß wären, überspringt diese Kombination direkt.
            # zu groß: mehr als 0.29 zusätzlich
            # wegen 2/9 Fehlerrate maximal die von den 7/9 übrigen Kanten jeweils fehlen darf.
            if merged_sizes[s_center] + merged_sizes[b_center] > 1.29 * mean_ndgree[b_center_i]:
                continue
            for x in range(0,sol_len):
                if parents[x, s_center] == parents[x, b_center]:
                    connectivity[x] = 1
                else:
                    connectivity[x] = 0
            # Berechne Gewicht:
            mwc = mean_node_weight(s_center, connectivity, vertex_costs, parents, sizes, n)
            # Aktualisiere ggf. best-passenden Knoten und minimalen mwc
            if mwc == -1:
                continue
            if mwc < min_mwc:
                min_mwc = mwc
                best_fit = b_center

        # Verbinde das Cluster mit dem Cluster, das im Mittel für s_center am günstigsten ist.
        rem_union(s_center, best_fit, merged)
        # Wg. Rem: aktualisiere Größe direkt in Repräsentanten von später erneut betrachtetem best_fit
        merged_sizes[best_fit] += merged_sizes[s_center]
    return merged

@njit
def calculate_frequencies(s_center, ccs, merged, merged_sizes, mean_ndgree, parents, sol_len, frequency, big_border):
    # Gehe jede Lösung durch
    for x in range(sol_len):
        # und jedes große Cluster b_center
        for b_center_i in range(len(ccs)):
            b_center = ccs[b_center_i]
            # falls kein großes Cluster, weiter
            if merged_sizes[b_center] <= mean_ndgree[b_center_i] * big_border:
                continue
            # falls Clustergrößen inkompatibel wären, auch weiter
            if merged_sizes[s_center] + merged_sizes[b_center] > 1.29 * mean_ndgree[b_center_i]:
                continue
            # falls in Lösung s_center direkt mit b_center verbunden ist: erhöhe Häufigkeit; betrachte nächstes b_center.
            if parents[x, s_center] == parents[x, b_center]:
                frequency[b_center] += 1
                continue
            # ansonsten teste ob s_center mit irgendeinem anderen Knoten aus dem b_center-Cluster in merged verbunden ist:
            else:
                b_center_members = np.where(merged == b_center)[0]
                for i in b_center_members:
                    if parents[x, s_center] == parents[x, i]:
                        frequency[b_center] += 1
                        # sobald die erste Verbindung zu b_center gefunden wurde, brich Scan ab
                        break
    return frequency

@njit
def repair_merged_v5_rem(merged, merged_sizes, solution_costs, vertex_costs, parents, sizes, n, node_dgree, big_border):
    sol_len = len(solution_costs)
    ccs_mndgr = calculate_mean_nodedgr_nd(merged, merged_sizes, node_dgree)
    ccs = ccs_mndgr[0]
    mean_ndgree = ccs_mndgr[1]
    frequency = np.zeros(n, dtype=np.int8)

    for s_center_i in range(len(ccs)):
        # s_center soll klein genug sein
        s_center = ccs[s_center_i]
        if merged_sizes[s_center] > mean_ndgree[s_center_i] * big_border:
            continue
        # Array wieder leeren (nach jedem Befüllen)
        for i in range(n):
            frequency[i] = 0
        # Detektiere und verbinde "Mini-Cluster" (Wurzel des Clusters soll verbunden werden).
        frequency = calculate_frequencies(s_center, ccs, merged, merged_sizes, mean_ndgree, parents, sol_len, frequency, big_border)
        best_fit = np.argmax(frequency)
        # Verbinde das Cluster mit dem Cluster, das mit s_center am häufigsten verbunden ist.
        rem_union(s_center, best_fit, merged)
        # Wg. Rem: aktualisiere Größe direkt in Repräsentanten von später erneut betrachtetem best_fit
        merged_sizes[best_fit] += merged_sizes[s_center]
    return merged

@njit
def greedy_find_local_best(local_best, x, y, z, vertex_costs):
    cand = np.zeros(3, dtype=np.int64)
    cand[0] = local_best[x]
    cand[1] = local_best[y]
    cand[2] = local_best[z]
    costs = np.zeros(3, dtype=np.float64)
    costs[0] = vertex_costs[cand[0], x] + vertex_costs[cand[0], y] + vertex_costs[cand[0], z]
    costs[1] = vertex_costs[cand[1], x] + vertex_costs[cand[1], y] + vertex_costs[cand[1], z]
    costs[2] = vertex_costs[cand[2], x] + vertex_costs[cand[2], y] + vertex_costs[cand[2], z]

    best = np.argmin(costs)
    return cand[best]


def repair_merged_local(merged, merged_sizes, solution_costs, vertex_costs, parents, sizes, n, node_dgree):
    sol_len = len(solution_costs)
    big_ccs = get_cluster_centers_big(merged, merged_sizes, node_dgree, 0.3)
    small_ccs = get_cluster_centers_small(merged, merged_sizes, node_dgree, 0.3)
    second_big_cc = get_second_center(merged, big_ccs)
    # O(n * x log x), weil für jeden Knoten x Einträge sortiert werden
    # Optimierungsmöglichkeit: nur Spalten sortieren, deren Knoten Clusterwurzeln sind.
    cost_sorted = np.argsort(vertex_costs, axis=0)
    local_best = cost_sorted[0]
    local_worst = cost_sorted[sol_len-1]
    worst_vertex_costs = np.zeros(n, dtype=np.float64)
    for i in range(n):
        worst_vertex_costs[i] = vertex_costs[local_worst[i], i]

    for s_center in small_ccs.keys():
        # Detektiere und verbinde "Mini-Cluster" (Wurzel des Clusters soll verbunden werden);
        # Reparatur wird versucht, wenn die Größe des Clusters weniger als halb so groß ist wie der Knotengrad angibt, dh. die lokale Fehlerrate wäre bei über 50% in der Probleminstanz.
        best_fit = s_center
        min_s_cost = worst_vertex_costs[s_center]
        for b_center in big_ccs.keys():
            # Falls Cluster zusammen deutlich zu groß wären, überspringe diese Kombination direkt
            if merged_sizes[s_center] + merged_sizes[b_center] > 1.6 * node_dgree[b_center]:
                continue
            local_i = greedy_find_local_best(local_best, s_center, b_center, second_big_cc[b_center], vertex_costs)
            local_solution = parents[local_i]
            # Falls der Knoten in der (greedy) lokal besten Lösung mit einem der beiden Cluster-Repräsentanten verbunden ist, führe Union durch.
            if local_solution[s_center] == local_solution[b_center] or local_solution[s_center] == local_solution[second_big_cc[b_center]]:
                if vertex_costs[local_i, s_center] < min_s_cost:
                    best_fit = b_center
                    min_s_cost = vertex_costs[local_i, s_center]
        # Verbinde das Cluster mit dem Cluster, das lokal betrachtet die geringsten Knotenkosten einbrachte.
        union(s_center, best_fit, merged, merged_sizes)
    result = np.zeros((2,n), dtype=np.int64)
    result[0] = merged
    result[1] = merged_sizes
    return result

def calculate_mean_nodedgr(merged, merged_sizes, node_dgree):
    cluster_center_mnd = {}
    for i in range(len(merged)):
        if merged[i] in cluster_center_mnd:
            cluster_center_mnd[merged[i]] += node_dgree[i]
        else:
            cluster_center_mnd[merged[i]] = node_dgree[i]
    for cc in cluster_center_mnd.keys():
        cluster_center_mnd[cc] = cluster_center_mnd[cc] / merged_sizes[cc]
    return cluster_center_mnd

@njit
def calculate_mean_nodedgr_nd(merged, merged_sizes, node_dgree):
    cluster_centers = np.unique(merged)
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

@njit
def get_second_center_nd(merged, cluster_centers):
    second_cc = np.zeros(len(cluster_centers), dtype=np.int64)
    j = 0
    for center in cluster_centers:
        # Durchlaufe solange andere Knoten bis einer aus dem selben Cluster gefunden wurde
        for i in range(len(merged)):
            # nicht der selbe Knoten ist gesucht
            if i == center:
                continue
            # sondern der erste, der einen anderen Index hat aber den selben Eintrag:
            if merged[i] == merged[center]:
                second_cc[j] = i
                j += 1
                break
    return second_cc


def repair_merged_local_v2(merged, merged_sizes, solution_costs, vertex_costs, parents, sizes, n, node_dgree):
    sol_len = len(solution_costs)
    ccs = calculate_mean_nodedgr(merged, merged_sizes, node_dgree)
    second_big_cc = get_second_center(merged, ccs)
    # O(n * x log x), weil für jeden Knoten x Einträge sortiert werden
    # Optimierungsmöglichkeit: nur Spalten sortieren, deren Knoten Clusterwurzeln sind.
    cost_sorted = np.argsort(vertex_costs, axis=0)
    local_best = cost_sorted[0]
    local_worst = cost_sorted[sol_len-1]
    worst_vertex_costs = np.zeros(n, dtype=np.float64)
    for i in range(n):
        worst_vertex_costs[i] = vertex_costs[local_worst[i], i]

    for s_center in ccs.keys():
        # s_center soll klein genug sein
        if merged_sizes[s_center] > ccs[s_center] * 0.5:
            continue
        # Detektiere und verbinde "Mini-Cluster" (Wurzel des Clusters soll verbunden werden);
        # Reparatur wird versucht, wenn die Größe des Clusters weniger als halb so groß ist wie der Knotengrad angibt, dh. die lokale Fehlerrate wäre bei über 50% in der Probleminstanz.
        best_fit = s_center
        min_s_cost = worst_vertex_costs[s_center]
        for b_center in ccs.keys():
            # b_center soll groß genug sein
            if merged_sizes[b_center] <= ccs[b_center] * 0.5:
                continue
            # Falls Cluster zusammen deutlich zu groß wären, überspringe diese Kombination direkt
            if merged_sizes[s_center] + merged_sizes[b_center] > 1.5 * ccs[b_center]:
                continue
            local_i = greedy_find_local_best(local_best, s_center, b_center, second_big_cc[b_center], vertex_costs)
            local_solution = parents[local_i]
            # Falls der Knoten in der (greedy) lokal besten Lösung mit beiden Cluster-Repräsentanten verbunden ist, ist er ein Kandidat für Union.
            if local_solution[s_center] == local_solution[b_center] or local_solution[s_center] == local_solution[second_big_cc[b_center]]:
                # Falls die Knotenkosten dieser lokalen Lösung geringer sind als bisheriger Lösungen,
                if vertex_costs[local_i, s_center] < min_s_cost:
                    # aktualisiere besten "Union-Partner" und die minimal beobachteten Knotenkosten.
                    best_fit = b_center
                    min_s_cost = vertex_costs[local_i, s_center]
        # Verbinde das Cluster mit dem Cluster, das lokal betrachtet die geringsten Knotenkosten einbrachte.
        union(s_center, best_fit, merged, merged_sizes)
    result = np.zeros((2,n), dtype=np.int64)
    result[0] = merged
    result[1] = merged_sizes
    return result

def merged_to_file(solutions, costs, filename, missing_weight, n, x, n_merges):
    """
    A function to write the merged solution(s) to a file, named like the input instance ending with _merged.txt.
    """
    print_to = filename[:-4] + "_merged_v5.txt"
    cost_sorted_j = np.argsort(costs)
    with open(print_to, mode="a") as file:
        file.write("filename: %s \nmissing_weight: %f \nn: %d \nx (solutions merged): %d\nmerged solutions:\n" % (filename, missing_weight, n, x))
        for j in cost_sorted_j:
            file.write(f"costs: {costs[j]}\n")
            for i in range(0,n):
                file.write(f"{solutions[j, i]} ")

def merged_to_file_mini(solutions, filename, missing_weight, n):
    """
    A function to write the merged solution(s) to a file, named like the input instance ending with _merged.txt.
    """
    print_to = filename[:-4] + "_merged_mini.txt"
    with open(print_to, mode="a") as file:
        file.write("filename: %s \nmissing_weight: %f \nn: %d \nmerged solution:\n" % (filename, missing_weight, n))
        for i in range(0,n):
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
