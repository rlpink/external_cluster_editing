from union_find import *
from math import log
import sys
import numpy as np
from numba import njit, jit
from numpy import random as rand
from model_sqrt import *
from merging_methods_v4_slim import *
import csv
from uf_cluster_editing_v4 import *
import datetime
"""
This module is not a module. It serves only one purpose, to be executed by a lido3-shell file.
It starts experiments for the unionfind_cluster_editing() algorithm.
"""

filename = sys.argv[1]
path = sys.argv[2]
rep = sys.argv[3]
seed = np.int64(sys.argv[4])
yxz = filename[:-8]
instance = filename[:-4]
output_path = "data/" + instance + "/" + rep + "/"
n = int(yxz.split('x')[0]) * int(yxz.split('x')[1])
file = open(output_path + "seed.txt", mode = "a")
file.write(str(seed))
file.close()
rand.seed(seed)
start_time = datetime.datetime.now()
unionfind_cluster_editing(path, output_path, -1, n, 1000)
end_time = datetime.datetime.now()
print(end_time - start_time)