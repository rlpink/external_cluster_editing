from union_find import *
import random
from math import log
import sys
import numpy
from numba import jit
from numpy import random as rand

### Preprocessing ###

# Knotengrade berechnen je Knoten (Scan über alle Kanten)


# Sequentiell für alle Lösungen (alle UF-Strukturen gleichzeitig, oder zumindest so viele wie passen):
### Generate Solutions ###

# 2. Scan über alle Kanten: Je Kante samplen in UF-Strukturen

### Solution Assessment ###

# 3. Scan über alle Kanten: Kostenberechnung für alle Lösungen (Gesamtkosten und Clusterkosten)

### Solution Merge ###

# Mithilfe der Bewertungen/Kosten Lösungen sinnvoll mergen/reparieren
# Platzhalter: Beste Lösung direkt übernehmen

# Summe der Kosten von Lösungen mit "A und B sind in einer ZHK" (=(A,B) in Lösung) vs. summierte Kosten für "A und B sind in verschiedenen ZHK"-Lösungen. Wähle Kante dann zufällig mit Gewichtung durch die Kosten (bspw (A,B) e L* "kostet" 4930, (A,B) -e L* "kostet" nur 1320. Dann teile (0,1) in Annahmebereich (0, 4930 / (4930 + 1320)) und Ablehnungsbereich (4930 / (4930 + 1320), 1). Würfle gleichverteilte Zufallszahl aus (0,1) und je nach Ausprägung (kleiner/ größer als der Grenzwert) füge Kante hinzu oder nicht.

# Cliquen einzeln bewerten und vergleichen:
# Ähnlichkeitsvergleich zwischen "Positionen mit gleichen Einträgen" in "flachen" UF-Arrays?
