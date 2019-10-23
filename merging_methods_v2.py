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
    print_to = filename[:-4] + "_all_solutions.txt"
    file = open(print_to, mode="a")
    count = 1
    with file:
        file.write("filename: %s \nmissing_weight: %f \nn: %d\n" % (filename, missing_weight, n))
        for i in cost_sorted_i:
            file.write("%d. best solution with cost %f\n" % (count, solution_costs[i]))
            count += 1
            for j in range(0,n):
                file.write(f"{parents[i][j]} ")
            file.write("\n")


def weighted_decision(x, y, cluster_masks, f_vertex_costs, f_sizes, f_parents):
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
            # Falls beide Zähler != 0 berechne "normales" Gewicht
            cost_1 = sum_for_1/count_1
            if cost_1 > 0:
                result = cost_0/cost_1 -1
            elif cost_0 == 0:
                print("Warning: Both together and single get cost 0 - no decision made")
                result = 0
            else:
                # Ersatzweise ein Trick, um das Verhältnis darzustellen
                result = (cost_0 + 0.1) / (cost_1 + 0.1) - 1

        else:
            # Falls kein Eintrag 1 gehört Knoten recht sicher nicht zum Cluster
            result = -1
    else:
        # Falls kein Eintrag 0 gehört Knoten recht sicher zum Cluster
        result = 1

    # Alle Fallunterscheidungen fangen sehr unwahrscheinliche Fälle ab; für genügend viele Lösungen sollte mit passender Samplingrate(!) keiner dieser Fälle eintreten!

    # Falls Rückgabe positiv: Entscheidung für 1, falls negativ: Entscheidung für 0.
    # Je näher Rückgabewert an 0, desto unsicherer die Entscheidung.
    return result


# c_opt: bester Modell-Parameter (mit geringsten Kosten, zB. durch meiste in Top-10) zw. 1-35
def merged_solution(solution_costs, vertex_costs, parents, sizes, filename, missing_weight, n):
    sol_len = len(solution_costs)

    # Neue Lösung als Array anlegen:
    merged_sol = np.arange(n, dtype=np.int64)
    merge_weight = np.zeros(n, dtype=float)

    # Arrays anlegen für Vergleichbarkeit der Cluster:
    cluster_masks = np.zeros((sol_len,n), dtype=np.int64)

    for j in rand.permutation(np.arange(0,n)):
        # Weise der Wurzel des Clusters ein hohes Gewicht für die "Entscheidung" zu:
        merge_weight[j] = 1
        # Prüfe, ob der Knoten noch nicht einem anderen Cluster zugeordnet wurde (Wurzeln werden nur 1x besucht)
        if merged_sol[j] == j:
            # Fülle Cluster-Masken
            for i in range(0,sol_len):
                # Jede Cluster-Maske enthält 1en überall, wo parents
                # denselben Wert hat wie an Stelle j
                for k in range(0,n):
                    if parents[i][k] == parents[i][j]:
                        cluster_masks[i][k] = 1
                    else:
                        cluster_masks[i][k] = 0
            # Berechne Zugehörigkeit zu Cluster (bzw. oder Nicht-Zugehörigkeit)
            for k in range(0,n):
                if k == j:
                    continue
                wd = weighted_decision(j, k, cluster_masks, vertex_costs, sizes, parents)
                # Falls neues Gewicht aussagekräftiger als voriges:
                if wd > merge_weight[k]:
                    # Ordne Knoten nach gewichteter Entscheidung diesem Cluster, j, zu
                    merged_sol[k] = j
    return merged_sol


def calc_sizes(solution):
    n = len(solution)
    sol_sizes = np.zeros(n, dtype=np.int64)
    for i in range(0, n):
        for j in range(0,n):
            if solution[i] == solution[j]:
                sol_sizes[i] += 1
    return sol_sizes


def merged_to_file(solutions, costs, filename, missing_weight, n, x, n_merges):
    print_to = filename[:-4] + "_merged.txt"
    file = open(print_to, mode="a")
    with file:
        file.write("filename: %s \nmissing_weight: %f \nn: %d \nx (solutions merged): %d\nmerged solutions:\n" % (filename, missing_weight, n, x))
        for j in range(n_merges):
            file.write(f"costs: {costs[j]}\n")
            for i in range(0,n):
                file.write(f"{solutions[j][i]} ")