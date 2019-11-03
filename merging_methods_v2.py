from union_find import *
from math import log
import sys
import numpy as np
from numba import njit, jit
from numpy import random as rand
from model_sqrt import *
"""
This module implements several methods for calculating and outputting solutions of the unionfind_cluster_editing() algorithm.
It contains two methods for the (best) generated raw solutions,
and, more importantly, methods to merge solutions into one better solution.
"""
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

def all_solutions(solution_costs, parents, filename, missing_weight, n, x):
    """
    This function outputs all solutions, sorted by their costs, to a ifle named "all_solutions.txt".
    """
    cost_sorted_i = np.argsort(solution_costs)
    print_to = filename[:-4] + "_all_solutions.txt"
    count = 1
    with open(print_to, mode="a") as file:
        file.write("filename: %s \nmissing_weight: %f \nn: %d\n" % (filename, missing_weight, n))
        for i in cost_sorted_i:
            file.write("%d. best solution with cost %f\n" % (count, solution_costs[i]))
            count += 1
            for j in range(0,n):
                file.write(f"{parents[i][j]} ")
            file.write("\n")


def weighted_decision(x, y, cluster_masks, f_vertex_costs, f_sizes, f_parents):
    """
    This function is a helper function for merging functions. It generates a weight for cluster center x and another node y by counting the costs over all solutions for two scenarios:
    1: y is in the same cluster as x
    0: y is in another cluster
    """
    sol_len = len(f_parents)
    sum_for_0 = 0
    sum_for_1 = 0
    count_0 = 0
    count_1 = 0
    for i in range(0,sol_len):
        x_cost = f_vertex_costs[i][x]
        y_cost = f_vertex_costs[i][y]
        if cluster_masks[i][y] == 0:
            sum_for_0 += x_cost + y_cost
            count_0 += 1
        else:
            sum_for_1 += x_cost + y_cost
            count_1 += 1
    # Solange nichts bekannt: Gewicht für Clusterzugehörigkeit 0 (gehört weder dazu noch nicht dazu, maximal Unsicherheit)
    result = 0
    if count_0 > 0:
        cost_0 = sum_for_0/count_0
        if count_1 > 0:
            cost_1 = sum_for_1/count_1
            if cost_0 == 0 and cost_1 == 0:
                print("Warning: Both together and single get cost 0 - no decision made")
                result = 0
            else:
                result = (cost_0 - cost_1) / (cost_0 + cost_1)

        else:
            # Falls kein Eintrag 1 gehört Knoten recht sicher nicht zum Cluster
            result = -1
    else:
        # Falls kein Eintrag 0 gehört Knoten recht sicher zum Cluster
        result = 1

    # Alle Fallunterscheidungen fangen sehr unwahrscheinliche Fälle ab; für genügend viele Lösungen sollte mit passender Samplingrate(!) keiner dieser Fälle eintreten!

    # Falls Rückgabe positiv: Entscheidung für 1 (zusammen), falls negativ: Entscheidung für 0 (getrennt).
    # Je näher Rückgabewert an 0, desto unsicherer die Entscheidung.
    return result


# c_opt: bester Modell-Parameter (mit geringsten Kosten, zB. durch meiste in Top-10) zw. 0-34
def merged_solution(solution_costs, vertex_costs, parents, sizes, missing_weight, n):
    """
    First merge algorithm. It calculates cluster masks for each cluster center:
    True, if the node is in the same component with cluster center,
    False otherwise.
    For these cluster masks, for each cluster center x and each other node y a weighted decision value is calculated. Is this weight better than the previous one, y gets assigned to new cluster center x. X then gets the weight of the maximum weight over all y, except if that is lower than its previous weight. Tree-like structures can emerge in such cases. Those trees are not handled yet, however they indicate a conflict in the solution, as a node that is both child and parent belongs to two distinct clusters.
    """
    sol_len = len(solution_costs)

    # Neue Lösung als Array anlegen:
    merged_sol = np.arange(n, dtype=np.int64)
    merge_weight = np.zeros(n, dtype=float)

    # Arrays anlegen für Vergleichbarkeit der Cluster:
    cluster_masks = np.zeros((sol_len,n), dtype=np.bool)

    for j in np.arange(0,n):
        # Fülle Cluster-Masken
        for i in range(0,sol_len):
            # Jede Cluster-Maske enthält "True" überall, wo parents
            # denselben Wert hat wie an Stelle j, sonst "False"
            for k in range(0,n):
                cluster_masks[i][k] = (parents[i][k] == parents[i][j])

        # Berechne Zugehörigkeit zu Cluster (bzw. oder Nicht-Zugehörigkeit)
        # Initialisiere Wert für Cluster-Zentrum j:
        wd_j = 0
        # Alle vorigen Knoten waren schon als Zentrum besucht und haben diesen Knoten daher schon mit sich verbunden (bzw. eben nicht) - Symmetrie der Kosten!
        for k in range(j+1,n):
            # Cluster-Zentrum wird übersprungen (dh. verweist möglicherweise noch auf anderes Cluster!)
            if k == j:
                continue
            wd = weighted_decision(j, k, cluster_masks, vertex_costs, sizes, parents)
            # Falls neues Gewicht aussagekräftiger als voriges:
            if wd > merge_weight[k]:
                # Prüfe, ob die Knoten nicht bereits in einem Cluster liegen:
                if merged_sol[k] != merged_sol[j]:
                    # Ordne Knoten nach gewichteter Entscheidung diesem Zentrum zu
                    merged_sol[k] = j
                    # Aktualisiere außerdem Gewicht
                    merge_weight[k] = wd
                    # Aktualisiere ggf. maximales Gewicht als Gewicht für Cluster-Zentrum
                    if wd > wd_j:
                        wd_j = wd
        # Falls das neue Gewicht größer (besser) ist als das bisherige, trenne das Cluster-Zentrum von seinem alten Cluster ab, indem es auf sich selbst zeigt.
        if wd_j > merge_weight[j]:
            merge_weight[j] = wd_j
            merged_sol[j] = j
    return merged_sol


def merged_to_file(solutions, costs, filename, missing_weight, n, x, n_merges):
    """
    A function to write the merged solution(s) to a file, named like the input instance ending with _merged.txt.
    """
    print_to = filename[:-4] + "_merged.txt"
    cost_sorted_j = np.argsort(costs)
    with open(print_to, mode="a") as file:
        file.write("filename: %s \nmissing_weight: %f \nn: %d \nx (solutions merged): %d\nmerged solutions:\n" % (filename, missing_weight, n, x))
        for j in cost_sorted_j:
            file.write(f"costs: {costs[j]}\n")
            for i in range(0,n):
                file.write(f"{solutions[j][i]} ")