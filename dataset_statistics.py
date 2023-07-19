"""
Module for calculating hyperbolicity of datasets and graph types.
"""
import utils.hyperbolicity as hyperbolicity
import networkx as nx
import matplotlib.pyplot as plt
import utils.data_utils as data_utils
from ricci import EdgeCurvature

datasets = ["cora", "pubmed", "airport", "disease_lp", "hrg_n1000", "sbm", "enschede_road"]

calculate_hyp = True
calculate_dc_plot = True
calculate_deg = True
calculate_clustering = True
calculate_edge_curv = True

num_samples = 50000


reported_delta = {"cora": 11.0, "pubmed": 3.5, "airport": 1.0, "disease_lp": 0.0}

path = "C:\\Users\\bp_ms\\Projects\\hgcn\\data\\"
def load_graph(dataset):
    data_path = path + dataset
    data = data_utils.load_data_lp(dataset, use_feats=False, data_path=data_path)
    graph = nx.from_scipy_sparse_matrix(data['adj_train'])


    # relabel nodes, only important when generating grid
    relabel_dict = {}
    for i, n in enumerate(list(graph.nodes())):
        relabel_dict[n] = i
    graph = nx.relabel_nodes(graph, relabel_dict)

    return graph


def get_degree_dist(graph):
    degs = nx.degree_histogram(graph)

    # fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    
    avg = 0
    for i, d in enumerate(degs):
        avg += i * degs[i]
    avg = avg / graph.number_of_nodes()

    return avg, degs

def get_clustering(graph):
    clustering_dict = nx.clustering(graph)
    clustering = []
    for node in clustering_dict:
        clustering.append(clustering_dict[node])
    return sum(clustering) / len(clustering), clustering

def get_dc_pairs(graph):
    """
    Get pairs (degree, clustering coefficient) for each node.
    """
    degs = []
    clusters = []
    for node in graph.nodes:
        d = graph.degree(node)
        c = nx.clustering(graph, node)
        degs.append(d)
        clusters.append(c)
    return degs, clusters


def make_big_table():

    num_rows = len(datasets)
    num_cols = 7

    fig, axs = plt.subplots(max(num_rows, 2), num_cols, figsize=(num_rows * 2, num_cols * 2))

    axs[0, 2].set_title("$\delta$ sampling")
    axs[0, 3].set_title("Degree-clustering plot")
    axs[0, 4].set_title("Degree distribution\n(log scale)")
    axs[0, 5].set_title("Clustering distribution")
    axs[0, 6].set_title("Edge curvature\ndistribution")

    for i, dataset in enumerate(datasets):
        print(f"Starting {dataset}")
        graph = load_graph(dataset)

        # precalculate everything
        if calculate_hyp:
            hyp, hyp_history = hyperbolicity.hyperbolicity_sample(graph, num_samples=num_samples)
        else:
            hyp = "-"
            hyp_history = None
        reported_hyp = reported_delta[dataset] if dataset in reported_delta else "-"
        if calculate_deg:
            avg_deg, degs = get_degree_dist(graph)
            avg_deg = round(avg_deg, 2)
        else:
            avg_deg = "-"
            degs = None
        if calculate_edge_curv:
            avg_edge_curv, edge_curv_count = EdgeCurvature(graph).get_curvature()
            avg_edge_curv = round(avg_edge_curv, 2)
        else:
            avg_edge_curv = "-"
            edge_curv_count = None
        if calculate_clustering:
            avg_clust, clustering = get_clustering(graph)
            avg_clust = round(avg_clust, 2)
        else:
            avg_clust = "-"
            clustering = None
        if calculate_dc_plot:
            dc_deg, dc_clust = get_dc_pairs(graph)
        else:
            dc_deg, dc_clust = None

        # dataset name
        axs[i, 0].text(0.5, 0.5, dataset, fontsize=12, ha='center')
        axs[i, 0].axis('off')

        # statistics
        cell_text = f"Sampled $\delta$: {hyp}\n" \
                    f"Reported $\delta$: {reported_hyp}\n" \
                    f"avg. deg.: {avg_deg}\n" \
                    f"avg. clustering: {avg_clust}\n" \
                    f"avg. edge curv.: {avg_edge_curv}"
        axs[i, 1].text(0, 0.25, cell_text, fontsize=9, ha='left')
        axs[i, 1].axis('off')

        # hyperbolicity history
        if hyp_history is not None:
            cell_text = ""
            for step in hyp_history:
                cell_text += f"\nIter.: {step[0]}, value {step[1]}"
            cell_text += f"\n Total iter: {num_samples}"
            axs[i, 2].text(0.5, 0.5, cell_text, fontsize=9, ha='center')
        axs[i, 2].axis('off')

        # d-c plot
        if dc_deg is not None:
            axs[i, 3].scatter(dc_deg, dc_clust, c='r', marker='.', s=1)
            axs[i, 3].set_ylim([0, 1])

        # degree distribution
        if degs is not None:
            # axs[i, 3].bar(range(len(degs)), degs)
            # axs[i, 3].set_xlabel('Degree')
            # axs[i, 3].set_ylabel('Count')
            
            axs[i, 4].bar(range(len(degs)), degs)
            # axs[i, 4].set_xlabel('Degree')
            # axs[i, 4].set_ylabel('Count')
            axs[i, 4].set_yscale('log')
        else:
            axs[i, 4].axis('off')
        
        # clustering
        if clustering is not None:
            axs[i, 5].hist(clustering)
            # axs[i, 5].set_xlabel('Clustering coeff.')
            # axs[i, 5].set_ylabel('Count')
            axs[i, 5].set_xlim([0, 1])
        else:
            axs[i, 5].axis('off')
        
        # edge curvature
        if edge_curv_count is not None:
            axs[i, 6].hist([edge_curv_count[i] for i in edge_curv_count])
            # axs[i, 6].set_xlabel("Edge curvature")
            # axs[i, 6].set_ylabel("Number of edges")
            axs[i, 6].set_xlim([-2, 3])
        else:
            axs[i, 6].axis('off')

    fig.tight_layout()
    plt.savefig("results\\big_table.png")

make_big_table()