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

###############################################################

method = "basic" # tsne or cosne or basic

experiment_id = "lp\\2023_6_8\\89" # {nc or lp}\\{date}\\{number of experiment}
n_dims = 2
dataset_name = "sbm"
use_embeddings = True # if false then just use original features for plot colors

sample_nodes = None # only take first n nodes
use_graph = False # whether to plot the graph or just the nodes

###############################################################

y = None

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
elif dataset_name in ["sbm"]:
    # colors_path = os.path.join("data", dataset_name, "labels.npy")
    graph_path = os.path.join("data", "sbm", "sbm.txt")
    adjacency_matrix = np.loadtxt(graph_path)
    num_nodes = adjacency_matrix.shape[0]

    # Create an empty graph adjacency dictionary
    adj = {}

    # Iterate over the adjacency matrix and populate the adjacency dictionary
    for node in range(num_nodes):
        neighbors = []
        for neighbor in range(num_nodes):
            if adjacency_matrix[node, neighbor] == 1:
                neighbors.append(neighbor)
        adj[node] = neighbors
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
if y is not None:
    palette = {0: 'r', 1: 'b', 2: 'g', 3: 'y', 4: 'c', 5: 'm', 6: 'tab:brown'}
    new_y = []
    for c in y:
        new_y.append(palette[c])
    y = np.array(new_y)

# run TSNE or not
if method == "tsne":
    tsne = TSNE(n_dims)
    tsne_result = tsne.fit_transform(X)
elif method == "cosne":
    raise Exception("COSNE not implemented")
elif method == "basic":
    tsne_result = X
else:
    raise Exception("Invalid method")



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
plt.show()