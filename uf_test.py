import numpy as np
from numba import njit, jit
from numpy import random as rand
from union_find import *
import pytest
from itertools import chain
import datetime
import pandas as pd
"""
This module contains some basic tests for correctness of the union_find module.
"""

def test_find():
    n = 10000
    uf = initialize_union_find(n)
    parent = uf[0]
    x = rand.randint(0,n)
    y = find(x, parent)
    assert y == x

def test_union():
    n = 10000
    uf = initialize_union_find(n)
    parent = uf[0]
    size = uf[1]
    x = rand.randint(0,n)
    y = rand.randint(0,n)
    x_size = size[x]
    y_size = size[y]
    union(x,y,parent, size)
    assert parent[x] == parent[y]
    assert size[parent[x]] == (x_size + y_size)

def test_flattening_find():
    n = 10000
    #artificial parent structure: tree is one giant branch, chained from 0 to n
    parent = np.arange(1,n+1)
    parent[n-1] = n-1
    flattening_find(0, parent)
    root = parent[0]
    #the root attribute - is 0 attached to a root?
    assert root == parent[root]
    #the root is the last element, namely n-1.
    assert root == (n-1)
    #is every element on the path directly attached to the root?
    for i in range(0, n):
        assert parent[i] == root


def test_rem_union():
    np.random.seed(123)
    set_time = True
    for r in range(100):
        start_time = datetime.datetime.now()
        parent = np.arange(10000)
        for i in range(25000):
            xy = np.random.randint(10000, size=2)
            rem_union(xy[0], xy[1], parent)
        for i in range(10000):
            parent[i] = flattening_find(i, parent)
        end_time = datetime.datetime.now()
        if set_time:
            time_sum = end_time-start_time
        else:
            time_sum += end_time-start_time
    print(time_sum / 100)
    return(parent)

def test_union2():
    np.random.seed(123)
    set_time = True
    for r in range(100):
        start_time = datetime.datetime.now()
        parent = np.arange(10000)
        size = np.ones(10000)
        for i in range(25000):
            xy = np.random.randint(10000, size=2)
            union(xy[0], xy[1], parent, size)
        for i in range(10000):
            parent[i] = flattening_find(i, parent)
        end_time = datetime.datetime.now()
        if set_time:
            time_sum = end_time-start_time
        else:
            time_sum += end_time-start_time
    print(time_sum / 100)
    return(parent)

def test_flattening_find_size():
        parent = np.arange(50)
        size = np.zeros(50)
        for i in range(20):
            xy = np.random.randint(50, size=2)
            rem_union(xy[0], xy[1], parent)
        for i in range(50):
            root = flattening_find(i, parent)
            size[root] += 1
        print(np.unique(parent))
        print(sum(size))
        print(size)

def test_rem_equals_union():
    for r in range(500):
        parent = np.arange(10000)
        parent2 = np.arange(10000)
        size = np.ones(10000)
        size2 = np.zeros(10000)
        for i in range(25000):
            xy = np.random.randint(10000, size=2)
            union(xy[0], xy[1], parent, size)
            rem_union(xy[0], xy[1], parent2)
        for i in range(10000):
            parent[i] = flattening_find(i, parent)
            parent2[i] = flattening_find(i, parent2)
            size2[parent2[i]] += 1
        roots = pd.unique(parent)
        n_clusts = len(roots)
        roots2 = pd.unique(parent2)
        n_clusts2 = len(roots2)
        assert n_clusts == n_clusts2
        for i in range(n_clusts):
            assert size[roots[i]] == size2[roots2[i]]
            c_root = roots[i]
            c_root2 = roots2[i]
            for j in range(10000):
                if parent[j] == c_root:
                    assert parent2[j] == c_root2
                if parent2[j] == c_root2:
                    assert parent[j] == c_root

