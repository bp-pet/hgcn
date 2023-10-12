from data.lfr_benchmark.lfr_benchmark import LFR_benchmark_graph as LFR
import matplotlib.pyplot as plt
import networkx as nx

# graph = LFR(n=250, tau1=3, tau2=1.5, mu=0.1, average_degree=5, min_community=20, seed=10)
graph = LFR(n=250, tau1=2, tau2=1.1, mu=0.1, min_degree=20, max_degree=50, seed=10)

nodes = graph.nodes(data=True)

communities = {frozenset(graph.nodes[v]["community"]) for v in graph}

colors = ['r', 'g', 'b', 'y', 'm', 'c', 'orange', 'brown']
# colors = ['r'] * 200
community_colors = {}
for i, c in enumerate(communities):
    community_colors[c] = colors[i]

node_colors = []
for node in nodes:
    node_colors.append(community_colors[frozenset(node[1]["community"])])

pos = nx.spring_layout(graph)

nx.draw_networkx_nodes(graph, pos, alpha=1, node_color=node_colors, node_size=15)
nx.draw_networkx_edges(graph, pos, alpha=0.2)

plt.axis("off")
plt.savefig("results\\datasets\\lfr.png", bbox_inches='tight', dpi=300)