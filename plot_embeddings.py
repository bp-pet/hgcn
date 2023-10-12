"""
Plot embeddings, either with t-SNE or CO-SNE, or just first two dimensions.
Last one is particularly useful when the embedding dimension is 2.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import os
import pickle as pkl
import utils.data_utils as data_utils


class EmbeddingPlotter:

    def __init__(self):
        self.adj = None
        self.features = None
        self.labels = None
        self.r = None # for HRG

        self.method = "basic"
        self.sample_nodes = None
        self.experiment_id = None

        with open("curvature\\last_experiment_id.txt", 'r') as f:
            self.experiment_id = f.read()
        
        self.experiment_id = None

        self.n_dims = 2
        self.use_graph = True

        self.noise = 0.0
        self.T = 0.0

    def load_data(self, dataset_name):

        print("Starting data loading")

        if dataset_name in ["cora"]:
            # premade by me from the existing data
            colors_path = os.path.join("data", dataset_name, "labels.npy")
            graph_path = os.path.join("data", dataset_name, "graph.pkl")
            self.labels = np.load(colors_path)
            with open(graph_path, 'rb') as f:
                self.adj = pkl.load(f)
        elif dataset_name[:3] in ["hrg"]:
            # temperature is loaded through the name, noise is added manually here
            # use the data utils function for loading, then change the format
            self.adj, self.features, self.labels = data_utils.load_hrg_data(dataset_name)
            self.labels = None  # no labels

            if self.noise is not None:
                self.features = data_utils.add_random_noise(self.features, self.noise, dataset_name)

            self.r = np.max(np.abs(self.features))

            # change to dict format
            adj_new = {}
            for i in range(self.adj.shape[0]):
                neighbors = []
                for j in range(self.adj.shape[1]):
                    if self.adj[i, j]:
                        neighbors.append(j)
                adj_new[i] = neighbors
            self.adj = adj_new
        elif dataset_name in ["sbm"]:
            # colors_path = os.path.join("data", dataset_name, "labels.npy")
            graph_path = os.path.join("data", "sbm", "sbm.txt")
            adjacency_matrix = np.loadtxt(graph_path)
            num_nodes = adjacency_matrix.shape[0]

            # Create an empty graph adjacency dictionary
            self.adj = {}

            # Iterate over the adjacency matrix and populate the adjacency dictionary
            for node in range(num_nodes):
                neighbors = []
                for neighbor in range(num_nodes):
                    if adjacency_matrix[node, neighbor] == 1:
                        neighbors.append(neighbor)
                self.adj[node] = neighbors
        elif dataset_name == "lfr":
            self.adj, self.features = data_utils.load_lfr_data(n=250)
        else:
            raise Exception("Need to specify paths and types for given dataset.")


        if self.experiment_id is not None:
            # load embeddings from experiment
            embeddings_path = os.path.join(self.experiment_id, "embeddings.npy")
            self.features = np.load(embeddings_path)
            print(self.features.shape)

        if self.sample_nodes is not None:
            X = X[:self.sample_nodes, :]
            labels = labels[:self.sample_nodes]
            # currently graph not used when sampling
            self.use_graph = False

        # replace class with color
        if self.labels is not None:
            palette = {0: 'r', 1: 'b', 2: 'g', 3: 'y', 4: 'c', 5: 'm', 6: 'tab:brown'}
            new_y = []
            for c in labels:
                new_y.append(palette[c])
            self.labels = np.array(new_y)

        # run TSNE or not
        if self.method == "tsne":
            tsne = TSNE(self.n_dims)
            self.features = tsne.fit_transform(self.features)
        elif self.method == "cosne":
            raise Exception("COSNE not implemented")
        elif self.method == "basic":
            pass
        else:
            raise Exception("Invalid method")


    def make_plot(self):
        print("Staring plot making")

        plt.clf()

        if self.n_dims == 2:
            features_df = pd.DataFrame({'tsne_1': self.features[:,0], 'tsne_2': self.features[:,1], 'label': self.labels})
            if self.use_graph:
                G = nx.from_dict_of_lists(self.adj)
                print(self.features.shape)
                nx.draw(G, self.features, node_size=30, node_color='r', edge_color='k')
            else:
                plt.scatter(self.features[:,0], self.features[:,1], c='r', s=5, zorder=2)

            #     for head_index in self.adj:
            #         head_row = features_df.iloc[[head_index]]
            #         head_coords = (head_row["tsne_1"].item(), head_row["tsne_2"].item())
            #         for tail_index in self.adj[head_index]:
            #             tail_row = features_df.iloc[[tail_index]]
            #             tail_coords = (tail_row["tsne_1"].item(), tail_row["tsne_2"].item())

            #             plt.plot((head_coords[0], tail_coords[0]),
            #                     (head_coords[1], tail_coords[1]),
            #                     c='b', alpha=0.05, zorder=1)

        elif self.n_dims == 3:
            # for 3 dim don't plot edges
            # features_df = pd.DataFrame({'tsne_1': features[:,0], 'tsne_2': features[:,1],
            #                                'tsne_3': features[:,2], 'label': y})
            # ax = fig.add_subplot(projection='3d')
            # ax.scatter(features[:,0], features[:,1], features[:,2], c=y)
            pass

        if self.r is not None:
            plt.xlim([-self.r, self.r])
            plt.ylim([-self.r, self.r])
        # plt.axis('off')
        plt.tight_layout()
        plt.axis('equal')


    def make_noise_plots(self):
        for i in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, -1.0]:
            print(i)
            self.noise = i
            self.load_data("hrg_n100")
            self.make_plot()
            if i == -1.0:
                i = "rand"
            plt.savefig(f"results\\noise\\hrg_{i}.png")
        self.noise = 0.0


        for i in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            print(i)
            self.T = i
            self.load_data(f"hrg_n100_t{i}")
            self.make_plot()
            if i == 1.0:
                i = "rand"
            plt.savefig(f"results\\noise\\hrg_t{i}.png")

if __name__ == "__main__":
    # EmbeddingPlotter().make_noise_plots()
    e = EmbeddingPlotter()

    e.load_data(f"hrg_n100")
    e.make_plot()

    plt.savefig(f"results\\datasets\\hrg.png", dpi=500)
    plt.show()