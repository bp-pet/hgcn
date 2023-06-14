"""
Module for calculating hyperbolicity of datasets and graph types.
"""
import os

import utils.hyperbolicity as hyperbolicity
import pickle as pkl
import sys
import time

import networkx as nx
import numpy as np
from tqdm import tqdm
import pandas as pd

import matplotlib.pyplot as plt

from utils.data_utils import load_data_lp, load_data_nc, load_hrg_data, load_lfr_data, load_sbm_data

from data.lfr_benchmark.lfr_benchmark import LFR_benchmark_graph

from ricci import RicciCurvature

def load_graph_from_csv(path):
    df = pd.read_csv(path, names=["source", "target"])
    g = nx.from_pandas_edgelist(df)
    return g


path = "C:\\Users\\bp_ms\\Projects\\hgcn\\data\\"
dataset = 'hrg'
data_path = path + dataset
data = load_data_lp(dataset, use_feats=False, data_path=data_path)
graph = nx.from_scipy_sparse_matrix(data['adj_train'])

# graph = load_graph_from_csv(path + "anybeat\\anybeat.csv")

# graph = nx.complete_graph(10)
# graph = nx.path_graph(10)
# graph = nx.cycle_graph(10)
# graph = nx.balanced_tree(2, 3)
# graph = nx.grid_2d_graph(5, 5)
# graph = nx.barabasi_albert_graph(100, 1)
# graph = nx.powerlaw_cluster_graph(10, 2, 0.1)i
# graph = nx.triangular_lattice_graph(5, 3)
# graph = nx.erdos_renyi_graph(10, 0.2)
# graph = nx.chordal_cycle_graph(7)
# graph = nx.stochastic_block_model([100, 100, 100], [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
# graph = LFR_benchmark_graph(800, 2, 1.1, 0.3, average_degree=6, seed=10)
# graph = nx.Graph(load_hrg_data()[0])
# graph = nx.Graph(load_lfr_data()[0])
# graph = nx.Graph(load_sbm_data()[0])


relabel_dict = {}
for i, n in enumerate(list(graph.nodes())):
    relabel_dict[n] = i
graph = nx.relabel_nodes(graph, relabel_dict)


print(f"Number of connected components: {nx.number_connected_components(graph)}")

print('Computing hyperbolicity', graph.number_of_nodes(), graph.number_of_edges())
hyp = hyperbolicity.hyperbolicity_sample(graph, num_samples=500000)
# hyp = hyperbolicity.hyperbolicity_full(graph)
print('Hyp: ', hyp)
# ricci = RicciCurvature(graph)
# print('Ricci:', ricci.get_graph_curvature())
# ricci.draw_with_curvature()



# nx.draw(graph)
# plt.suptitle(dataset)
# plt.savefig(f"curvhist_{dataset}.png")