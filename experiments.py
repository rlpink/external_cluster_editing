"""
This module is not a module. It serves only one purpose, to be executed by a lido3-shell file.
It starts experiments for the unionfind_cluster_editing() algorithm.
"""
from union_find import *
from math import log
import sys
import numpy as np
from numba import njit, jit
from numpy import random as rand
from model_sqrt import *
from merging_methods_v5 import *
import csv
from uf_cluster_editing_v5 import *
import datetime


filename = sys.argv[1]
path = sys.argv[2]
rand.seed(1234)
yxz = filename[:-8]
n = int(yxz.split('x')[0]) * int(yxz.split('x')[1])
start_time = datetime.datetime.now()
print(start_time)
unionfind_cluster_editing(path, -1, n, 1000, 1)
end_time = datetime.datetime.now()
print(end_time)
print(end_time - start_time)
