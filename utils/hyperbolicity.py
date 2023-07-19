import os
import pickle as pkl
import sys
import time

import networkx as nx
import numpy as np
from tqdm import tqdm
import itertools



def hyperbolicity_sample(G, num_samples=50000):
    curr_time = time.time()

    history = []
    current_max = 0
    for i in tqdm(range(num_samples)):
        node_tuple = np.random.choice(G.nodes(), 4, replace=False)
        try:
            hyp = calculate_hyp(G, node_tuple[0], node_tuple[1], node_tuple[2], node_tuple[3])
            if hyp > current_max:
                history.append((i, hyp))
                current_max = hyp
        except Exception as e:
            continue
    print('Time for hyp: ', time.time() - curr_time)
    return current_max, history

def calculate_hyp(G, a, b, c, d):
        s = []
        d01 = nx.shortest_path_length(G, source=a, target=b, weight=None)
        d23 = nx.shortest_path_length(G, source=c, target=d, weight=None)
        d02 = nx.shortest_path_length(G, source=a, target=c, weight=None)
        d13 = nx.shortest_path_length(G, source=b, target=d, weight=None)
        d03 = nx.shortest_path_length(G, source=a, target=d, weight=None)
        d12 = nx.shortest_path_length(G, source=b, target=c, weight=None)
        s.append(d01 + d23)
        s.append(d02 + d13)
        s.append(d03 + d12)
        s.sort()
        return (s[-1] - s[-2]) / 2

def hyperbolicity_full(G):
    curr_time = time.time()
    hyps = []
    for node_tuple in tqdm(list(itertools.product(list(G.nodes()), repeat=4))):
        try:
            hyps.append(calculate_hyp(G, node_tuple[0], node_tuple[1], node_tuple[2], node_tuple[3]))
        except Exception as e:
            continue
    print('Time for hyp: ', time.time() - curr_time)
    return max(hyps)
