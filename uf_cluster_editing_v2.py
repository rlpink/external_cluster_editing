from union_find import *
from math import log
import sys
import numpy as np
from numba import njit, jit
from numpy import random as rand
from model_sqrt import *
from merging_methods_v2 import *
import csv

# Input sollte aus je 3 mit Leerzeichen getrennten Einträgen pro Zeile bestehen:
# <Nummer Knoten 1> <Nummer Knoten 2> <Gewicht der Kante>
# Die Knotenbezeichnungen sind (von 0 bis n-1) numpy.int64, die Gewichte numpy.float64

# missing_weight: Gewicht für fehlende Kanten (die nicht in der Datei enthalten sind)
# n: Anzahl Objekte/Knoten
# x: Anzahl generierter Lösungen (mehr = besser, aber teurer in Speicher/Laufzeit)
# zu x kommen noch n_merges generierte Merge-Lösungen hinzu, sollte im Budget berücksichtigt werden

def unionfind_cluster_editing(filename, missing_weight, n, x, n_merges):
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
    sizes = np.ones((x,n), dtype=np.int64)
    # Modellparameter einlesen:
    parameters_b = load_model_flexible_v2('params_below_100.csv')
    parameters_a = load_model_flexible_v2('params_above_100.csv')
    #cluster_count = np.full(x, n, dtype=np.int64)
    # Alle Parameter für die Modelle festlegen:
    cluster_model = np.full(x,17)
    def generate_solutions(first, c_opt):
        if first:
            k = int(x/35)
            j = 0
            c = 0

            for i in range(0,x):
                cluster_model[i] = c
                j += 1
                if j == k and c < 34:
                    c += 1
                    j = 0
        if not first:
            # Überschreibe Lösungen mit nicht-optimalem Parameter um danach neue zu generieren
            for i in range(0,x):
                if cluster_model[i] != c_opt:
                    parents[i] = np.arange(n, dtype=np.int64)
                    sizes[i] = np.ones(n, dtype = np.int64)

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
                    union(nodes[0], nodes[1], parents[i], sizes[i])

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

        for i in range(0,x):
            if merged and cluster_model[i] == c_opt:
                # Ändere in 2. Lauf nichts an den Lösungen, die bereits gut sind!
                continue
            parent_uf = solutions_parents[i]
            size_uf = inner_sizes[i]
            for j in range(0,n):
                root = flattening_find(j,parent_uf)
                n_c = inner_sizes[i][root]
                c_edge_counter[i][j] = n_c - 1

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
                if merged and cluster_model[i] == c_opt:
                    # Ändere in 2. Lauf nichts an den Lösungen, die bereits gut sind!
                    continue
                if not merged:
                    root1 = find(nodes[0],solutions_parents[i])
                    root2 = find(nodes[1],solutions_parents[i])
                else:
                    root1 = solutions_parents[i][nodes[0]]
                    root2 = solutions_parents[i][nodes[1]]
                # Kante zwischen zwei Clustern
                if root1 != root2:
                    # mit positivem Gewicht (zu viel)
                    if weight > 0:
                        vertex_costs[i][nodes[0]] += weight / 2
                        vertex_costs[i][nodes[1]] += weight / 2
                        solution_costs[i] += weight
                # Kante innerhalb von Cluster
                else:
                    # mit negativem Gewicht (fehlt)
                    if weight < 0:
                        vertex_costs[i][nodes[0]] -= weight / 2
                        vertex_costs[i][nodes[1]] -= weight / 2
                        solution_costs[i] -= weight
                    c_edge_counter[i][nodes[0]] -= 1
                    c_edge_counter[i][nodes[1]] -= 1
                    #print("missing edges for now: ", c_edge_counter[i][root1])

        for i in range(0,x):
            # über Cluster(-Repräsentanten, Keys) iterieren:
            for j in range(n):
                missing_edges = c_edge_counter[i][j]
                if missing_edges > 0:
                    # Kosten für komplett fehlende Kanten zur Lösung addieren
                    vertex_costs[i][j] += missing_edges * (-missing_weight) * 0.5
                    solution_costs[i] += missing_edges * (-missing_weight) * 0.5 # Zwei Knoten innerhalb eines Clusters vermissen die selbe Kante, daher *0.5 bei Berechnung über die Knoten
        return (vertex_costs, solution_costs)
    costs = calculate_costs(parents, x, False)
    vertex_costs = costs[0]
    solution_costs = costs[1]

    all_solutions(solution_costs, parents, filename, missing_weight, n, x)

### Solution Merge ###

# Mithilfe der Bewertungen/Kosten Lösungen sinnvoll mergen/reparieren
# Platzhalter: Beste Lösung direkt übernehmen
    print("begin solution merge")
    #best_solution(solution_costs, parents, filename, missing_weight, n, x)
    #all_solutions(solution_costs, parents, filename, missing_weight, n, x)

    mean_costs_c = np.zeros(35, dtype=np.float64)
    k = int(x/35)
    rest = x % 35
    # Summierte Kosten für selben Parameter
    for i in range(x):
        c = cluster_model[i]
        mean_costs_c[c] = mean_costs_c[c] + solution_costs[i]
    # Teilen durch Anzahl Lösungen mit dem Parameter (k oder k+rest für 35)
    for i in range(35):
        if i == 34:
            mean_costs_c[i] = mean_costs_c[i] / (k + rest)
        else:
            mean_costs_c[i] = mean_costs_c[i] / k
    # c_opt ist Parameter mit geringsten Durchschnittskosten der Lösungen
    c_opt = np.argsort(mean_costs_c)[0] + 1 # Rückrechnung Index zu Parameter

    generate_solutions(False, c_opt)
    costs = calculate_costs(parents, x, False)
    vertex_costs = costs[0]
    solution_costs = costs[1]

    # merged_solutions = np.full((n_merges,n), np.arange(n, dtype=np.int64))
    # merged_sizes = np.full((n_merges,n), np.zeros(n, dtype=np.int64))

   ##   for i in range(0,n_merges):
    #     merged = merged_solution(solution_costs, vertex_costs, parents, sizes, missing_weight, n)
    #     print("generated merge: ", i)
    #     merged_solutions[i] = merged
    #     merged_sizes[i] = calc_sizes(merged)
    merged = merged_solution(solution_costs, vertex_costs, parents, sizes, missing_weight, n)
    merged_costs = calculate_costs(merged_solutions, n_merges, True)[1]
    merged_to_file(merged_solutions, merged_costs, filename, missing_weight, n, x, n_merges)
    all_solutions(solution_costs, parents, filename, missing_weight, n, x)
