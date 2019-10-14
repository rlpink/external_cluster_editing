from union_find import *
from math import log
import sys
import numpy as np
from numba import njit, jit
from numpy import random as rand
from model_sqrt import *
from merging_methods import *

# Input sollte aus je 3 mit Leerzeichen getrennten Einträgen pro Zeile bestehen:
# <Nummer Knoten 1> <Nummer Knoten 2> <Gewicht der Kante>
# Die Knotenbezeichnungen sind (von 0 bis n-1) numpy.int64, die Gewichte numpy.float64

# missing_weight: Gewicht für fehlende Kanten (die nicht in der Datei enthalten sind)
# n: Anzahl Objekte/Knoten
# x: Anzahl generierter Lösungen (mehr = besser, aber teurer in Speicher/Laufzeit)

def unionfind_cluster_editing(filename, missing_weight, n, x):
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
    cluster_count = np.full(x, n, dtype=np.int64)
    #model_quantile = np.full(x, 0.5)

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
        # Samplingrate ermitteln
        sampling_rate = model_flexible(guess_n, 0.9)

        decision_values = rand.rand(x)
        for i in range(0, x):
            # Falls Kante gesamplet...
            # sampling_rate = model_sqrt(guess_n, model_quantile[i])
            if decision_values[i] < sampling_rate:
                # ...füge Kante ein in UF-Struktur
                # Falls "echte" Vereinigung (zwei vorher verschiedene Cluster)...
                if union(nodes[0], nodes[1], parents[i], sizes[i]):
                    #...reduziere Anzahl Cluster in der Lösung:
                    cluster_count[i] = cluster_count[i] - 1


### Solution Assessment ###
# Nachbearbeitung aller Lösungen: Flache Struktur (= Knoten in selbem Cluster haben selben Eintrag im Array)
# Und Berechnung benötigter Kanten je Cluster (n_c * (n_c-1) / 2) pro UF
    print("begin solution assessment")
    solution_costs = np.zeros(x, dtype=np.float64)

    #todo: dict(dict(...)) oder np.array(dict(...))?
    #cluster_costs = dict()
    cluster_costs = np.empty(x, dtype='O')
    c_edge_counter = np.empty(x, dtype='O')
    # Ende von todo

    for i in range(0,x):
        cluster_costs[i] = dict()
        c_edge_counter[i] = dict()
        parent_uf = parents[i]
        size_uf = sizes[i]
        for j in range(0,n):
            root = flattening_find(j,parent_uf)
            cluster_costs[i][root] = np.float64(0)
            n_c = sizes[i][root]
            c_edge_counter[i][root] = np.int64((n_c * (n_c-1)) / 2)


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
            root1 = find(nodes[0],parents[i])
            root2 = find(nodes[1],parents[i])
            # Kante zwischen zwei Clustern
            if root1 != root2:
                # mit positivem Gewicht (zu viel)
                if weight > 0:
                    cluster_costs[i][root1] += weight / 2
                    cluster_costs[i][root2] += weight / 2
                    solution_costs[i] += weight
            # Kante innerhalb von Cluster
            else:
                # mit negativem Gewicht (fehlt)
                if weight < 0:
                    cluster_costs[i][root1] -= weight
                    solution_costs[i] -= weight
                c_edge_counter[i][root1] -= 1

    for i in range(0,x):
        # über Cluster(-Repräsentanten, Keys) iterieren:
        for c in c_edge_counter[i]:
            missing_edges = c_edge_counter[i][c]
            if missing_edges > 0:
                # Kosten für komplett fehlende Kanten zur Lösung addieren
                cluster_costs[i][c] += missing_edges * (-missing_weight)
                solution_costs[i] += missing_edges * (-missing_weight)


### Solution Merge ###

# Mithilfe der Bewertungen/Kosten Lösungen sinnvoll mergen/reparieren
# Platzhalter: Beste Lösung direkt übernehmen
    print("begin solution merge")
    #best_solution(solution_costs, parents, filename, missing_weight, n, x)
    all_solutions(solution_costs, parents, filename, missing_weight, n, x)

# Summe der Kosten von Lösungen mit "A und B sind in einer ZHK" (=(A,B) in Lösung) vs. summierte Kosten für "A und B sind in verschiedenen ZHK"-Lösungen. Wähle Kante dann zufällig mit Gewichtung durch die Kosten (bspw (A,B) e L* "kostet" 4930, (A,B) -e L* "kostet" nur 1320. Dann teile (0,1) in Annahmebereich (0, 4930 / (4930 + 1320)) und Ablehnungsbereich (4930 / (4930 + 1320), 1). Würfle gleichverteilte Zufallszahl aus (0,1) und je nach Ausprägung (kleiner/ größer als der Grenzwert) füge Kante hinzu oder nicht.

# Cliquen einzeln bewerten und vergleichen:
# Ähnlichkeitsvergleich zwischen "Positionen mit gleichen Einträgen" in "flachen" UF-Arrays?
