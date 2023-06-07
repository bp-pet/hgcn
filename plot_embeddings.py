"""
Plot embeddings, either with t-SNE or CO-SNE, or just first two dimensions.
Last one is particularly useful when the embedding dimension is 2.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import os
import pickle as pkl
import torch
try:
    from cosne.main import run_TSNE as run_COSNE
    from cosne.main import plot_low_dims as plot_COSNE
except:
    print("warning: loading cosne failed")
    pass

###############################################################

method = "basic" # tsne or cosne or basic

# experiment_id = "lp\\2023_4_18\\35" # {nc or lp}\\{date}\\{number of experiment}
experiment_id = "lp\\2023_6_7\\1" # {nc or lp}\\{date}\\{number of experiment}
n_dims = 2 # should be 2
dataset_name = "hrg"
use_embeddings = True # if false then just use original features for plot colors

sample_nodes = None # only take first n nodes
use_graph = True # whether to plot the graph or just the nodes

###############################################################

# load colors and graph from dataset
if dataset_name in ["cora"]:
    colors_path = os.path.join("data", dataset_name, "labels.npy")
    graph_path = os.path.join("data", dataset_name, "graph.pkl")
    y = np.load(colors_path)
    with open(graph_path, 'rb') as f:
        adj = pkl.load(f)
elif dataset_name in ["hrg"]:
    from utils.data_utils import load_hrg_data
    adj, X, y = load_hrg_data()

    # change to dict format
    adj_new = {}
    for i in range(adj.shape[0]):
        neighbors = []
        for j in range(adj.shape[1]):
            if adj[i, j]:
                neighbors.append(j)
        adj_new[i] = neighbors
    adj = adj_new
else:
    raise Exception("Need to specify paths and types for given dataset.")


if use_embeddings:
    # load embeddings from experiment
    embeddings_path = os.path.join("logs", experiment_id, "embeddings.npy")
    X = np.load(embeddings_path)

if sample_nodes is not None:
    X = X[:sample_nodes, :]
    y = y[:sample_nodes]
    # currently graph not used when sampling
    use_graph = False

# replace class with color
palette = {0: 'r', 1: 'b', 2: 'g', 3: 'y', 4: 'c', 5: 'm', 6: 'tab:brown'}
new_y = []
for c in y:
    new_y.append(palette[c])
y = np.array(new_y)

# run TSNE
if method == "tsne":
    tsne = TSNE(n_dims)
    tsne_result = tsne.fit_transform(X)
elif method == "cosne":
    pass
    # tsne_result = run_COSNE(torch.Tensor(X), only_cosne=True)
elif method == "basic":
    tsne_result = X
else:
    raise Exception("Invalid method")

print(X)

# plot result
# fig = plt.figure()

if n_dims == 2:
    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y})
    # ax = fig.add_subplot()
    if use_graph:
        for head_index in adj:
            head_row = tsne_result_df.iloc[[head_index]]
            head_coords = (head_row["tsne_1"].item(), head_row["tsne_2"].item())
            for tail_index in adj[head_index]:
                tail_row = tsne_result_df.iloc[[tail_index]]
                tail_coords = (tail_row["tsne_1"].item(), tail_row["tsne_2"].item())

                plt.plot((head_coords[0], tail_coords[0]),
                        (head_coords[1], tail_coords[1]),
                        c='k', alpha=0.1, zorder=1)
    plt.scatter(tsne_result[:,0], tsne_result[:,1], c=y, s=2, zorder=2)

elif n_dims == 3:
    # for 3 dim don't plot edges
    # tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1],
    #                                'tsne_3': tsne_result[:,2], 'label': y})
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(tsne_result[:,0], tsne_result[:,1], tsne_result[:,2], c=y)
    pass
plt.title(method)
# ax.set_aspect('equal')
plt.savefig(f"images\\{method}_{dataset_name}.png", dpi=1000)