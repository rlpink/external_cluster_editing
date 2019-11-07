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

def print_solution_costs(solution_costs, filename):
    """
    This function outputs all sorted solution costs to a ifle named "..._solution_costs.txt".
    """
    sorted_costs = np.sort(solution_costs)
    print_to = filename[:-4] + "_solution_costs.txt"
    with open(print_to, mode="a") as file:
        for cost in sorted_costs:
            file.write(str(cost))
            file.write("\n")

def all_solutions(solution_costs, parents, filename, missing_weight, n):
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
        x_cost = f_vertex_costs[i][x]
        y_cost = f_vertex_costs[i][y]
        if cluster_masks[i][y] == 0:
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
    return 0


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
    merged_sizes = np.ones(n, dtype=np.int64)

    # Arrays anlegen für Vergleichbarkeit der Cluster:
    cluster_masks = np.zeros((sol_len,n), dtype=np.int8) #np.bool not supported

    for j in range(0,n):
        # Fülle Cluster-Masken
        for i in range(0,sol_len):
            # Jede Cluster-Maske enthält "True" überall, wo parents
            # denselben Wert hat wie an Stelle j, sonst "False"
            for k in range(0,n):
                cluster_masks[i][k] = np.int8(parents[i][k] == parents[i][j])

        # Berechne Zugehörigkeit zu Cluster (bzw. oder Nicht-Zugehörigkeit)
        # Alle vorigen Knoten waren schon als Zentrum besucht und haben diesen Knoten daher schon mit sich verbunden (bzw. eben nicht) - Symmetrie der Kosten!
        for k in range(j+1,n):
            # Cluster-Zentrum wird übersprungen (dh. verweist möglicherweise noch auf anderes Cluster!)
            if k == j:
                continue
            wd = weighted_decision(j, k, cluster_masks, vertex_costs, sizes, parents)
            # Falls Gewicht groß genug:
            if wd > 0.05:
                union(j, k, merged_sol, merged_sizes)
    result = np.zeros((2,n))
    result[0] = merged_sol
    result[1] = merged_sizes

    return result

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