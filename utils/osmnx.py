"""
For making networkx graph of map data. Might not work with default hgcn venv
as it is missing osmnx.
"""
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import pandas as pd


coords = (52.2209855, 6.8940537)

dist = 5000

G = ox.graph_from_point(coords, dist=dist, network_type="drive")
G = nx.Graph(G)
relabel_dict = {}
for i, n in enumerate(list(G.nodes())):
    relabel_dict[n] = i
G = nx.relabel_nodes(G, relabel_dict)

adj_dict = {node: tuple(G.neighbors(node)) for node in G.nodes}
with open("enschede_adj.txt", 'w') as f:
    for k in adj_dict:
        f.write(str(k))
        f.write(" ")
        for l in adj_dict[k]:
            f.write(str(l))
            f.write(" ")
        f.write("\n")

node_positions_dict = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes}
with open("enschede_pos.txt", 'w') as f:
    for k in node_positions_dict:
        f.write(str(k))
        f.write(" ")
        for l in node_positions_dict[k]:
            f.write(str(l))
            f.write(" ")
        f.write("\n")


# Plot the graph with node positions
# nx.draw_networkx(G, pos=node_positions, with_labels=False, node_size=1)
# plt.show()