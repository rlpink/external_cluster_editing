from union_find import *
import random
from math import log
import sys
import numpy as np
from numba import jit
from numpy import random as rand

def model_sqrt(n):
    # Parameters fitted for 99.9% quantile of necessary edges for connectivity
    a = 3.484291
    b =  -2.983427
    c = -12.853034
    edges = (0.5 * n * np.log(n)) + (a * n) + b * np.sqrt(n) + c
    max_edges = n * (n-1) / 2

    return edges / max_edges

# Input sollte aus je 3 mit Leerzeichen getrennten Einträgen pro Zeile bestehen:
# <Nummer Knoten 1> <Nummer Knoten 2> <Gewicht der Kante>
# Die Knotenbezeichnungen sind (von 0 bis n-1) numpy.int64, die Gewichte numpy.float64

# missing_weight: Gewicht für fehlende Kanten (die nicht in der Datei enthalten sind)
# n: Anzahl Objekte/Knoten
# x: Anzahl generierter Lösungen (mehr = besser, aber teurer in Speicher/Laufzeit)

def unionfind_cluster_editing(filename, missing_weight, n, x):
    graph_file = open(filename, mode="r")

### Preprocessing ###

# Knotengrade berechnen je Knoten (Scan über alle Kanten)
    node_dgr = np.zeros(n, dtype=np.int64)

    for line in graph_file:
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
    parents = np.full((x,n), np.arange(n, dtype=np.int64))
    sizes = np.ones((x,n), dtype=np.int64)

# 2. Scan über alle Kanten: Je Kante samplen in UF-Strukturen
    graph_file = open(filename, mode="r")

    for line in graph_file:
        splitted = line.split()
        nodes = np.array(splitted[:-1], dtype=np.int64)
        weight = np.float64(splitted[2])
        guess_n = (node_dgr[nodes[0]] + node_dgr[nodes[1]]) / 2
        sampling_rate = model_sqrt(guess_n)

        decision_values = rand.rand(x)
        for i in range(0, x):
            # Falls Kante gesamplet...
            if decision_values[i] < sampling_rate:
                # ...füge Kante ein in UF-Struktur
                union(nodes[0], nodes[1], parents[i], sizes[i])


### Solution Assessment ###

# 3. Scan über alle Kanten: Kostenberechnung für alle Lösungen (Gesamtkosten und Clusterkosten)
    graph_file = open(filename, mode="r")

    for line in graph_file:


### Solution Merge ###

# Mithilfe der Bewertungen/Kosten Lösungen sinnvoll mergen/reparieren
# Platzhalter: Beste Lösung direkt übernehmen

# Summe der Kosten von Lösungen mit "A und B sind in einer ZHK" (=(A,B) in Lösung) vs. summierte Kosten für "A und B sind in verschiedenen ZHK"-Lösungen. Wähle Kante dann zufällig mit Gewichtung durch die Kosten (bspw (A,B) e L* "kostet" 4930, (A,B) -e L* "kostet" nur 1320. Dann teile (0,1) in Annahmebereich (0, 4930 / (4930 + 1320)) und Ablehnungsbereich (4930 / (4930 + 1320), 1). Würfle gleichverteilte Zufallszahl aus (0,1) und je nach Ausprägung (kleiner/ größer als der Grenzwert) füge Kante hinzu oder nicht.

# Cliquen einzeln bewerten und vergleichen:
# Ähnlichkeitsvergleich zwischen "Positionen mit gleichen Einträgen" in "flachen" UF-Arrays?
