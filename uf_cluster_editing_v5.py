from union_find import *
from math import log
import sys
import numpy as np
from numba import njit, jit
from numpy import random as rand
from model_sqrt import *
from merging_methods_v5 import *
import csv

"""
This module implements a cluster editing algorithm. It uses a semi-streaming approach and is therefore able to process files that would be too big for main memory.
"""

# Input sollte aus je 3 mit Leerzeichen getrennten Einträgen pro Zeile bestehen:
# <Nummer Knoten 1> <Nummer Knoten 2> <Gewicht der Kante>
# Die Knotenbezeichnungen sind (von 0 bis n-1) numpy.int64, die Gewichte numpy.float64

# missing_weight: Gewicht für fehlende Kanten (die nicht in der Datei enthalten sind)
# n: Anzahl Objekte/Knoten
# x: Anzahl generierter Lösungen (mehr = besser, aber teurer in Speicher/Laufzeit)

def unionfind_cluster_editing(filename, missing_weight, n, x, n_merges):

    """
    This is a cluster editing algorithm, based on semi-streaming approach using union find to analyze graph structures.
    The input file should contain edges in format
    <np.int64: edge 1> <np.int64: edge 2> <np.float64: edge weight>\n
    Parameter missing_weight sets a weight for edges that are not contained in the file (for unweighted data: -1)
    Parameter n gives the number of objects (nodes)
    Parameter x is the number of generated solutions (which are the basis for a merged solution). It merely influences running time, however with limited memory it should not be chosen too high. 300-1k is recommended, the more the better.
    """
    graph_file = open(filename, mode="r")


### Preprocessing ###
    print("Begin preprocessing\n")
# Knotengrade berechnen je Knoten (Scan über alle Kanten)
    node_dgr = np.zeros(n, dtype=np.int64)

    for line in graph_file:
        # Kommentar-Zeilen überspringen
        if line[0] == "#":
            continue
        splitted = line.split()
        nodes = np.array(splitted[:-1], dtype=np.int64)
        weight = np.float64(splitted[2])

        # Falls Kante 'existiert' nach Threshold:
        if weight > 0:
            node_dgr[nodes[0]] += 1
            node_dgr[nodes[1]] += 1

# Sequentiell für alle Lösungen (alle UF-Strukturen gleichzeitig, oder zumindest so viele wie passen):
# Größe einer Lösung: Array mit n Einträgen aus je 64bit
### Generate Solutions ###
    print("begin solution generation")
    parents = np.full((x,n), np.arange(n, dtype=np.int64))
    sizes = np.zeros((x,n), dtype=np.int64)
    # Modellparameter einlesen:
    parameters_b = load_model_flexible_v2('params_below_100.csv')
    parameters_a = load_model_flexible_v2('params_above_100.csv')
    #cluster_count = np.full(x, n, dtype=np.int64)
    # Alle Parameter für die Modelle festlegen:
    cluster_model = np.full(x,17)
    def generate_solutions(first, c_opt):
        if first:
            k = int(x/37)
            j = 0
            c = 0

            for i in range(0,x):
                cluster_model[i] = c
                j += 1
                if j == k and c < 36:
                    c += 1
                    j = 0
        if not first:
            # Überschreibe Lösungen mit nicht-optimalem Parameter um danach neue zu generieren
            for i in range(0,x):
                if cluster_model[i] != c_opt:
                    parents[i] = np.arange(n, dtype=np.int64)
                    sizes[i] = np.zeros(n, dtype = np.int64)

    # 2. Scan über alle Kanten: Je Kante samplen in UF-Strukturen
        graph_file = open(filename, mode="r")

        for line in graph_file:
            # Kommentar-Zeilen überspringen
            if line[0] == "#":
                continue
            splitted = line.split()
            nodes = np.array(splitted[:-1], dtype=np.int64)
            weight = np.float64(splitted[2])

            guess_n = (node_dgr[nodes[0]] + node_dgr[nodes[1]]) / 2

            decision_values = rand.rand(x)
            for i in range(0, x):
                if not first:
                    if cluster_model[i] == c_opt:
                        # Ändere in 2. Lauf nichts an den Lösungen, die bereits gut sind!
                        continue
            # Samplingrate ermitteln
                sampling_rate = model_flexible_v2(parameters_b, parameters_a, guess_n, cluster_model[i])
                # Falls Kante gesamplet...
                if decision_values[i] < sampling_rate:
                    # ...füge Kante ein in UF-Struktur
                    rem_union(nodes[0], nodes[1], parents[i])

    generate_solutions(True, 0)



### Solution Assessment ###
# Nachbearbeitung aller Lösungen: Flache Struktur (= Knoten in selbem Cluster haben selben Eintrag im Array)
# Und Berechnung benötigter Kanten je Cluster (n_c * (n_c-1) / 2) pro UF

    def calculate_costs(solutions_parents, x, merged):
        if merged:
            inner_sizes = merged_sizes
        else:
            inner_sizes = sizes
        print("begin solution assessment")
        solution_costs = np.zeros(x, dtype=np.float64)
        vertex_costs = np.zeros((x,n), dtype=np.float64)
        c_edge_counter = np.zeros((x,n), dtype=np.int64)

        for i in range(x):
            for j in range(n):
                root = flattening_find(j,solutions_parents[i])
                inner_sizes[i, root] += 1
            for j in range(n):
                root = solutions_parents[i, j]
                n_c = inner_sizes[i, root]
                c_edge_counter[i, j] = n_c - 1

        # 3. Scan über alle Kanten: Kostenberechnung für alle Lösungen (Gesamtkosten und Clusterkosten)
        graph_file = open(filename, mode="r")

        for line in graph_file:
            # Kommentar-Zeilen überspringen
            if line[0] == "#":
                continue
            splitted = line.split()
            nodes = np.array(splitted[:-1], dtype=np.int64)
            weight = np.float64(splitted[2])

            for i in range(0,x):
                if not merged:
                    root1 = find(nodes[0],solutions_parents[i])
                    root2 = find(nodes[1],solutions_parents[i])
                else:
                    root1 = solutions_parents[i, nodes[0]]
                    root2 = solutions_parents[i, nodes[1]]
                # Kante zwischen zwei Clustern
                if root1 != root2:
                    # mit positivem Gewicht (zu viel)
                    if weight > 0:
                        vertex_costs[i, nodes[0]] += weight / 2
                        vertex_costs[i, nodes[1]] += weight / 2
                        solution_costs[i] += weight
                # Kante innerhalb von Cluster
                else:
                    # mit negativem Gewicht (fehlt)
                    if weight < 0:
                        vertex_costs[i, nodes[0]] -= weight / 2
                        vertex_costs[i, nodes[1]] -= weight / 2
                        solution_costs[i] -= weight
                    c_edge_counter[i, nodes[0]] -= 1
                    c_edge_counter[i, nodes[1]] -= 1
                    #print("missing edges for now: ", c_edge_counter[i][root1])

        for i in range(0,x):
            # über Cluster(-Repräsentanten, Keys) iterieren:
            for j in range(n):
                missing_edges = c_edge_counter[i, j]
                if missing_edges > 0:
                    # Kosten für komplett fehlende Kanten zur Lösung addieren
                    vertex_costs[i, j] += missing_edges * (-missing_weight) * 0.5
                    solution_costs[i] += missing_edges * (-missing_weight) * 0.5 # Zwei Knoten innerhalb eines Clusters vermissen die selbe Kante, daher *0.5 bei Berechnung über die Knoten
        return (vertex_costs, solution_costs)
    costs = calculate_costs(parents, x, False)
    vertex_costs = costs[0]
    solution_costs = costs[1]

### Solution Merge ###

# Mithilfe der Bewertungen/Kosten Lösungen sinnvoll mergen/reparieren
# Platzhalter: Beste Lösung direkt übernehmen
    print("begin solution merge")

    mean_costs_c = np.zeros(37, dtype=np.float64)
    c_count = np.zeros(37, dtype= np.int64)
    # Summierte Kosten für selben Parameter
    for i in range(x):
        c = cluster_model[i]
        mean_costs_c[c] = mean_costs_c[c] + solution_costs[i]
        c_count[c] += 1
    # Teilen durch Anzahl Lösungen mit dem Parameter
    for i in range(37):
        mean_costs_c[i] = mean_costs_c[i]/c_count[i]
    # c_opt ist Parameter mit geringsten Durchschnittskosten der Lösungen
    c_opt = np.argsort(mean_costs_c)[0]

    generate_solutions(False, c_opt)
    costs = calculate_costs(parents, x, False)
    vertex_costs = costs[0]
    solution_costs = costs[1]
    # Optimierung: Filtern der "besten" Lösungen, um eine solidere Basis für den Merge zu schaffen.
    # best_costs_i = np.argmin(solution_costs)
    # best_costs = solution_costs[best_costs_i]
    # good_costs_i = np.where(solution_costs <= best_costs * 1.7)
    # Variante 2: Beste 10% verwenden
    top_percent = range(np.int64(x*0.1))
    mid_percent = range(np.int64(x*0.90))
    cost_sorted_i = np.argsort(solution_costs)
    good_costs_i = cost_sorted_i[top_percent]
    mid_costs_i = cost_sorted_i[mid_percent]
    # Artefakt aus Zeit mit n_merges > 1; sonst inkompatibel mit calculate_costs.
    merged_solutions = np.full((n_merges,n), np.arange(n, dtype=np.int64))
    merged_sizes = np.full((n_merges,n), np.zeros(n, dtype=np.int64))
    for i in range(0,n_merges):
        merged = merged_solution(solution_costs[good_costs_i], vertex_costs[good_costs_i], parents[good_costs_i], sizes[good_costs_i], missing_weight, n)
        merged_solutions[i] = merged[0]
        merged_sizes[i] = merged[1]
        # Glätten der Lösung falls Baumstruktur auftritt
        for j in range(0,n):
            r = flattening_find(j, merged_solutions[i])
            merged_sizes[i, r] += 1
        merged_costs = calculate_costs(merged_solutions, n_merges, True)[1]
        rep = repair_merged_v4_nd_rem(merged_solutions[i], merged_sizes[i], solution_costs[mid_costs_i], vertex_costs[mid_costs_i], parents[mid_costs_i], sizes[mid_costs_i], n, node_dgr, 0.3)
        merged_solutions[i] = rep
        merged_sizes[i] = np.zeros(n, dtype=np.int64)
    merged_costs = calculate_costs(merged_solutions, n_merges, True)[1]
    # Da Merge auf x2 Lösungen basiert, nur diese angeben:
    x2 = len(good_costs_i)
    merged_to_file(merged_solutions, merged_costs, filename, missing_weight, n, x2, n_merges)
    #all_solutions(solution_costs[good_costs_i], parents[good_costs_i], filename, missing_weight, n)
    merged_short_print(merged_solutions, merged_costs, filename, missing_weight, n, x2, n_merges)