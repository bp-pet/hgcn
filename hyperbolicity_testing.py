import utils.hyperbolicity as hyperbolicity
import os
import pickle as pkl
import sys
import time

import networkx as nx
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from utils.data_utils import load_data_lp


path = "C:\\Users\\bp_ms\\Projects\\hgcn\\data\\"
dataset = 'cora'
data_path = path + dataset
data = load_data_lp(dataset, use_feats=False, data_path=data_path)
graph = nx.from_scipy_sparse_matrix(data['adj_train'])

# graph = nx.complete_graph(10)
# graph = nx.path_graph(10)
# graph = nx.cycle_graph(10)
# graph = nx.grid_2d_graph(5, 5)


relabel_dict = {}
for i, n in enumerate(list(graph.nodes())):
    relabel_dict[n] = i
graph = nx.relabel_nodes(graph, relabel_dict)


print(f"Number of connected components: {nx.number_connected_components(graph)}")

# print('Computing hyperbolicity', graph.number_of_nodes(), graph.number_of_edges())
hyp = hyperbolicity.hyperbolicity_sample(graph, num_samples=50000)
# hyp = hyperbolicity.hyperbolicity_full(graph)
print('Hyp: ', hyp)


# nx.draw(graph)
# plt.show()