import networkx as nx
from utils.data_utils import load_hrg_data
import matplotlib.pyplot as plt
from tqdm import tqdm





class EdgeCurvature:

    def __init__(self, graph):
        self.graph = graph

    def get_edge_curvature(self, graph, edge):
        i = edge[0]
        j = edge[1]

        
        d_i = graph.degree[i]
        d_j = graph.degree[j]

        triangles = self.get_num_triagles(i, j)
        squares_i, gamma = self.get_num_squares(i, j)
        squares_j, _ = self.get_num_squares(j, i)

        H1 = triangles / max(d_i, d_j)
        H2 = triangles / min(d_i, d_j)
        if squares_i == 0:
            H3 = 0
        else:
            H3 = (squares_i + squares_j) / (max(d_i, d_j) * gamma)

        return (2 / d_i) + (2 / d_j) - 2 + 2 * H1 + H2 + H3


    def get_num_triagles(self, i, j):
        return len(set(self.graph.neighbors(i)).intersection(set(self.graph.neighbors(j))))

    def get_num_squares(self, i, j):
        cycles = []
        neighbors_i = list(self.graph.neighbors(i))
        neighbors_j = list(self.graph.neighbors(j))
        for first_stop in neighbors_i:
            if first_stop in neighbors_j or first_stop == j:
                continue
            for second_stop in list(self.graph.neighbors(first_stop)):
                if second_stop in neighbors_i or second_stop == i:
                    continue
                if self.graph.has_edge(second_stop, j):
                    cycles.append((first_stop, second_stop))
        if len(cycles) == 0:
            return 0, 0
        else:
            return len(cycles), self.get_gamma(cycles)

    def get_gamma(self, cycles):
        occurences = {}
        for cycle in cycles:
            for node in cycle:
                if node in occurences:
                    occurences[node] += 1
                else:
                    occurences[node] = 1
        return (max(occurences[i] for i in occurences))


    def get_curvature(self, sample_num=None, hist_num_bins=None):
        """
        Set the curvature of each edge as an attribute and return the average curvature.
        """
        edges = list(self.graph.edges())
        result = 0
        curvature_dict = {}
        for e in tqdm(edges):

            # check for leaf
            i = e[0]
            j = e[1]
            d_i = self.graph.degree[i]
            d_j = self.graph.degree[j]
            if d_i == 1 or d_j == 1:
                # print("here")
                # curvature_dict[e] = round(c, 2)
                continue
            
            c = self.get_edge_curvature(self.graph, e)

            result += c
            curvature_dict[e] = round(c, 2)
        nx.set_edge_attributes(self.graph, curvature_dict, "curvature")
        avg = result / len(edges)
        return avg, curvature_dict
    
    def draw_with_curvature(self):
        edge_labels = nx.get_edge_attributes(self.graph, "curvature")
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos=pos, node_size=10)
        nx.draw_networkx_edge_labels(self.graph, pos=pos, edge_labels=edge_labels)
        # plt.savefig("images\\ricci.png")