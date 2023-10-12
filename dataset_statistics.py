"""
Module for calculating hyperbolicity of datasets and graph types.
"""
import utils.hyperbolicity as hyperbolicity
import networkx as nx
import matplotlib.pyplot as plt
import utils.data_utils as data_utils
from ricci import EdgeCurvature
import numpy as np

datasets = ["cora", "pubmed", "airport", "hrg_n1000", "sbm", "lfr", "disease_lp", "enschede_road"]
# datasets = ["hrg_n1000", "sbm"]
# datasets = ["cora", "pubmed", "airport", "disease_lp"]
# # datasets = ["enschede_road", "lfr"]
# datasets = ["hrg_n1000_a0.55", "hrg_n1000_a0.60", "hrg_n1000_a0.65", "hrg_n1000_a0.70", "hrg_n1000_a0.75",
#             "hrg_n1000_a0.80", "hrg_n1000_a0.85", "hrg_n1000_a0.90", "hrg_n1000_a0.95", "hrg_n1000_a1.00"]
# datasets = ["hrg_n1000_a0.60", "hrg_n1000_a0.75", "hrg_n1000_a0.90"]
# datasets = ["sbm"]
# datasets = ["enschede_road"]
datasets = ["disease_lp"]
# datasets = ["lfr"]

calculate_hyp = 1
calculate_dc_plot = 1
calculate_deg = 1
calculate_clustering = 1
calculate_edge_curv = 1

num_samples = 50000

bf_curv_upper_lim = 2.5
degree_upper_lim = 60

dcplot_ylim = [0, 1]
# dcplot_ylim = [-0.1, 0.9]

tick_size = 16


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

def get_hyperbolicity():
    for dataset in datasets:
        graph = load_graph(dataset)
        hyp = hyperbolicity.hyperbolicity_sample(graph, num_samples=num_samples)
        with open("results\\delta_hyp.txt", 'a') as f:
            f.write(f"\n{dataset}, {hyp}, {num_samples} samples")




def get_degree_dist(graph):
    """Retirn average degree, degree historgram, and estimated power of degree distribution.
    """
    deg_counts = nx.degree_histogram(graph)
    degs = list(range(len(deg_counts)))
    
    avg = 0
    for i, d in enumerate(deg_counts):
        avg += i * deg_counts[i]
    avg = avg / graph.number_of_nodes()

    # delete zeros
    x = np.array(degs[1:])
    y = np.array(deg_counts[1:])
    zero_indices = np.where(y == 0)[0]
    x = np.delete(x, zero_indices)
    y = np.delete(y, zero_indices)

    # log transform and find power
    log_x = np.log(x)
    log_y = np.log(y)

    # fit excluding bias (only k without C)
    # k = np.dot(log_x, log_y) / (np.linalg.norm(log_x) ** 2)

    # limit points for fit
    boundary1 = int(0.2 * x.shape[0])
    boundary2 = int(0.8 * x.shape[0])

    log_x = log_x[boundary1:boundary2]
    log_y = log_y[boundary1:boundary2]
    
    # polyfit
    coeff, res,_,_,_ = np.polyfit(log_x, log_y, 1, full=True)
    a, b = coeff[0], coeff[1]
    res = res.item()
    res = res / x.shape[0]
    C = np.exp(b)
    k = -a

    # plot fit
    # plt.clf()
    # plt.scatter(log_x, log_y)
    # plt.plot(log_x, a * log_x + b)
    # plt.scatter(x, y)
    # plt.plot(x, C * (x**(-k)))
    # plt.show()
    # print(C, k)
    # raise Exception

    return avg, x, y, C, k, res

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


def make_edge_curv_plots():
    # unmute to find maximum
    # abs_max = -2
    # for dataset in datasets:
    #     graph = load_graph(dataset)
    #     avg_edge_curv, edge_curv_count = EdgeCurvature(graph).get_curvature()
    #     edge_curv_count = list(edge_curv_count.values())
    #     dataset_max = max(edge_curv_count)
    #     if dataset_max > abs_max:
    #         abs_max = dataset_max
    # print(abs_max)
    # raise Exception
    
    for dataset in datasets:
        graph = load_graph(dataset)
        avg_edge_curv, edge_curv_count = EdgeCurvature(graph).get_curvature()
        edge_curv_count = list(edge_curv_count.values())

        print(edge_curv_count)
        raise Exception

        if max(edge_curv_count) > bf_curv_upper_lim:
            raise Exception("Found BF curvature above given limit")
        
        plt.clf()
        plt.hist(edge_curv_count, bins=20)
        plt.xlim([-2, bf_curv_upper_lim])

        plt.tick_params(axis='x', labelsize=tick_size)
        plt.tick_params(axis='y', labelsize=tick_size)

        plt.savefig(f"results\\bf_curvature\\curvhist_{dataset}.png")

        with open(f"results\\bf_curvature\\averages.txt", 'a') as f:
            f.write(f"\n{dataset}: {avg_edge_curv}")

def make_degree_plots():
    # unmute to find maximum
    # abs_max = 0
    # for dataset in datasets:
    #     graph = load_graph(dataset)
    #     avg_deg, degs, deg_counts, deg_power, deg_power_res = get_degree_dist(graph)
    #     dataset_max = degs[-1]
    #     if dataset_max > abs_max:
    #         abs_max = dataset_max
    # print(abs_max)
    # raise Exception
    
    for dataset in datasets:
        graph = load_graph(dataset)
        avg_deg, degs, deg_counts, deg_C, deg_power, deg_power_res = get_degree_dist(graph)

        print(f"C: {deg_C}, k: {-deg_power}, res: {deg_power_res}")

        this_max = degs[-1]

        if degs[-1] > degree_upper_lim:
            print(degs[-1])
            raise Exception("Degree found given limit")
        
        plt.clf()
        plt.scatter(degs, deg_counts)
        # plt.xscale('log')
        # plt.yscale('log')

        # plot fit
        # plt.plot(degs, deg_C * (degs ** (-deg_power)))
        
        plt.xlim([1, degree_upper_lim])
        # plt.xlim([0, this_max])

        plt.tick_params(axis='x', labelsize=tick_size)
        plt.tick_params(axis='y', labelsize=tick_size)

        # plt.show()

        plt.savefig(f"results\\degrees\\degdist_{dataset}.png")

        # with open(f"results\\degrees\\averages.txt", 'a') as f:
        #     f.write(f"\n{dataset}: {avg_deg}")

def make_clustering_plots():
    
    for dataset in datasets:
        graph = load_graph(dataset)
        avg_clust, clustering = get_clustering(graph)

        plt.clf()
        plt.hist(clustering, bins=20)
        plt.xlim([0, 1])

        plt.tick_params(axis='x', labelsize=tick_size)
        plt.tick_params(axis='y', labelsize=tick_size)

        plt.savefig(f"results\\clustering\\clustdist_{dataset}.png")

        with open(f"results\\clustering\\averages.txt", 'a') as f:
            f.write(f"\n{dataset}: {avg_clust}")

def get_density():
    for dataset in datasets:
        graph = load_graph(dataset)
        print(graph.number_of_nodes())
        density = nx.density(graph)
        with open(f"results\\degrees\\density.txt", 'a') as f:
            f.write(f"\n{dataset}: {density}")


def make_dc_plots():

    for dataset in datasets:
        graph = load_graph(dataset)
        deg, cl = get_dc_pairs(graph)

        plt.clf()
        plt.scatter(deg, cl, c='r', marker='.', s=50)
        plt.xlim([0, degree_upper_lim])
        plt.ylim(dcplot_ylim)

        plt.tick_params(axis='x', labelsize=tick_size)
        plt.tick_params(axis='y', labelsize=tick_size)

        plt.savefig(f"results\\dc_plots\\dcplot_{dataset}.png")


def make_big_table():

    num_rows = len(datasets)
    num_cols = 7

    fig, axs = plt.subplots(max(num_rows, 2), num_cols, figsize=(num_cols * 3, num_rows * 2))

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
            avg_deg, degs, deg_counts, deg_power, deg_power_res = get_degree_dist(graph)
            avg_deg = round(avg_deg, 2)
            deg_power = round(deg_power, 2)
            deg_power_res = round(deg_power_res, 2)
        else:
            avg_deg = "-"
            deg_counts = None
            deg_power = "-"
            deg_power_res = "-"
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
            dc_deg = None
            dc_clust = None

        # dataset name
        axs[i, 0].text(0.5, 0.5, dataset, fontsize=12, ha='center')
        axs[i, 0].axis('off')

        # statistics
        cell_text = f"Sampled $\delta$: {hyp}\n" \
                    f"Reported $\delta$: {reported_hyp}\n" \
                    f"avg. deg.: {avg_deg}\n" \
                    f"avg. clustering: {avg_clust}\n" \
                    f"avg. edge curv.: {avg_edge_curv}\n" \
                    f"deg. dist. power: {deg_power}, residual: {deg_power_res}"
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
        if deg_counts is not None:
            # axs[i, 3].bar(range(len(degs)), degs)
            # axs[i, 3].set_xlabel('Degree')
            # axs[i, 3].set_ylabel('Count')
            
            axs[i, 4].scatter(range(len(deg_counts)), deg_counts)
            # axs[i, 4].set_xlabel('Degree')
            # axs[i, 4].set_ylabel('Count')
            axs[i, 4].set_xscale('log')
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


if __name__ == "__main__":
    # make_big_table()
    make_edge_curv_plots()
    # make_degree_plots()
    # get_density()
    # make_clustering_plots()
    # make_dc_plots()
    # get_hyperbolicity()
    pass