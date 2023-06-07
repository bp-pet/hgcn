"""
Module for checking datatypes and shapes of the cora dataset.
"""

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd

# cora
# 2708 nodes, 7 classes


# graph: dict {index: [list of indices]}, 2708 keys
# test.index: txt file of 1000 indices

# allx: csr matrix 1708x1433, features
# ally: np array 1708x7, labels

# tx: csr matrix 1000x1433, features
# ty: np array 1000x7, labels

# x: csr matrix 140x1433, features; not used
# y: np array 140x7, labels; only use the len (140) to determine size of test set


# features: allx, tx
# labels: ally, ty
# order doesn't seem to matter

# graph: graph adjacency list

# test indices: y, order doesn't matter only size


file = "ind.cora.test.index"

with open(file, 'rb') as f:
    data = pkl.load(f, encoding='latin1')

print(type(data))
print(data.shape)
print(data.keys())