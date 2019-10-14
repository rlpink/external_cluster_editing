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
