from union_find import *
from math import log
import sys
import numpy as np
from numba import njit, jit
from numpy import random as rand
from model_sqrt import *

def best_solution(solution_costs, parents, filename, missing_weight, n, x):
    costs = solution_costs.min()
    best = parents[solution_costs.argmin()]
    file = open("result.txt", mode="a")
    with file:
        file.write("filename: %s \nmissing_weight: %f \nn: %d \nx (solutions generated): %d\nbest solution found:\n" % (filename, missing_weight, n, x))
        file.write(f"costs: {costs}\n")
        for i in range(0,n):
            file.write(f"{best[i]} ")

def all_solutions(solution_costs, parents, filename, missing_weight, n, x):
    cost_sorted_i = np.argsort(solution_costs)
    file = open("result.txt", mode="a")
    count = 1
    with file:
        file.write("filename: %s \nmissing_weight: %f \nn: %d\n" % (filename, missing_weight, n))
        for i in cost_sorted_i:
            file.write("%d. best solution with cost %f\n" % (count, solution_costs[i]))
            count += 1
            for j in range(0,n):
                file.write(f"{parents[i][j]} ")
            file.write("\n")
def mean_vertex_cost(vertex, solution, cluster_costs, sizes, parents):
    root = parents[solution][vertex]
    c_c = cluster_costs[solution][root]
    n_c = sizes[solution][root]
    return c_c/n_c

def weighted_decision(x, y, cluster_masks, f_cluster_costs, f_sizes, f_parents):
    sum_for_0 = 0
    sum_for_1 = 0
    count_0 = 0
    count_1 = 0
    for i in range(0,sol_len):
        x_mcost = mean_vertex_cost(x, i, f_cluster_costs, f_sizes, f_parents)
        y_mcost = mean_vertex_cost(y, i, f_cluster_costs, f_sizes, f_parents)
        if cluster_masks[i][y] = 0:
            sum_for_0 += x_mcost + y_mcost
            count_0 += 1
        else:
            sum_for_1 += x_mcost + y_mcost
            count_1 += 1
    if ((sum_for_0/count_0) - (sum_for_1/count_1)) < 0:
        return 0
    else:
        return 1

# c_opt: bester Modell-Parameter (mit geringsten Kosten, zB. durch meiste in Top-10) zw. 1-35
def merged_solution(solution_costs, cluster_costs, parents, sizes, filename, missing_weight, n, x, c_opt):
    best_i = np.where(cluster_model == c_opt)
    f_solution_costs = solution_costs[best_i]
    f_cluster_costs = cluster_costs[best_i]
    f_parents = parents[best_i]y_01,
    f_sizes = sizes[best_i]
    sol_len = len(f_solution_costs)

    # Neue Lösung als Array anlegen:
    merged_sol = np.arange(n, dtype=np.int64)

    # Arrays anlegen für Vergleichbarkeit der Cluster:
    cluster_masks = np.zeros((sol_len+1,n), dtype=np.int64)

    for j in rand.permutation(np.arange(0,n)):
        # Prüfe, ob der Knoten noch nicht einem anderen Cluster zugeordnet wurde (Wurzeln werden nur 1x besucht)
        if merged_sol[j] == j:
            # Fülle Cluster-Masken
            for i in range(0,sol_len):
                # Jede Cluster-Maske enthält 1en überall, wo f_parents
                # denselben Wert hat wie an Stelle j
                for k in range(0,n):
                    cluster_masks[i][np.where(f_parents[k] == f_parents[j])] = 1
            # Berechne Zugehörigkeit zu Cluster (bzw. oder Nicht-Zugehörigkeit)
            for k in range(0,n):
                if k == j:
                    continue
                wd = weighted_decision(j, k, cluster_masks, f_cluster_costs, f_sizes, f_parents)
                if wd == 0:
                    continue
                else:
                    # Ordne Knoten nach gewichteter Entscheidung diesem Cluster, j, zu
                    merged_sol[k] = j

    return merged_sol

