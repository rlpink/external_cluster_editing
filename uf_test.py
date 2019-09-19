import numpy as np
from numba import njit, jit
from numpy import random as rand
from union_find import *
import pytest
from itertools import chain

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

