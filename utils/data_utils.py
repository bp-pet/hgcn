"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys
import collections
import subprocess

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import snap
import matplotlib.pyplot as plt

from data.lfr_benchmark.lfr_benchmark import LFR_benchmark_graph as LFR


def load_data(args, datapath):
    if args.task == 'nc':
        data = load_data_nc(args.dataset, args.use_feats, datapath, args.split_seed)
    else:
        data = load_data_lp(args.dataset, args.use_feats, datapath)
        adj = data['adj_train']
        if args.task == 'lp':
            adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                    adj, args.val_prop, args.test_prop, args.split_seed
            )
            data['adj_train'] = adj_train
            data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
            data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
            data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
    
    data["features"] = add_random_noise(data["features"], args.noise_std, args.dataset)

    data['adj_train_norm'], data['features'] = process(
            data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
    )
    if args.dataset == 'airport':
        data['features'] = augment(data['adj_train'], data['features'])
    return data


# ############### FEATURES PROCESSING ####################################

def add_random_noise(data, std, dataset_name):
    """
    Add Gaussian noise to array with given standard deviation.
    """
    if std == 0.0:
        return data
    if dataset_name[:3] == "hrg":
        # scale std depending on max radius
        max_radius = np.sqrt(np.max(data[:, 0] ** 2 + data[:, 1] ** 2))
        if std < 0:
            noise = np.random.normal(0, -std * max_radius, data.shape)
            return noise
        else: 
            noise = np.random.normal(0, std * max_radius, data.shape)
            return data + noise
    else:
        raise Exception("Adding noise to given dataset not implemented")


def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


# ############### DATA SPLITS #####################################################


def mask_edges(adj, val_prop, test_prop, seed):
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false)  


def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


# ############### LINK PREDICTION DATA LOADERS ####################################


def load_data_lp(dataset, use_feats, data_path):
    if dataset in ['cora', 'pubmed']:
        adj, features = load_citation_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'disease_lp':
        adj, features = load_synthetic_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'airport':
        adj, features = load_data_airport(dataset, data_path, return_label=False)  
    elif dataset[:3] == 'hrg':
        adj, features = load_hrg_data(dataset, use_feats)[:2]
    elif dataset == 'lfr':
        adj, features = load_lfr_data()[:2]
    elif dataset == 'sbm':
        adj, features = load_sbm_data(deterministic=False)[:2]
    elif dataset == 'roadnet_ca':
        adj, features = load_roadnet_ca()
    elif dataset == 'enschede_road':
        adj, features = load_enschede_road()
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    data = {'adj_train': adj, 'features': features}
    return data


# ############### NODE CLASSIFICATION DATA LOADERS ####################################


def load_data_nc(dataset, use_feats, data_path, split_seed):
    if dataset in ['cora', 'pubmed']:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
            dataset, use_feats, data_path, split_seed
        )
    else:
        if dataset == 'disease_nc':
            adj, features, labels = load_synthetic_data(dataset, use_feats, data_path)
            val_prop, test_prop = 0.10, 0.60
        elif dataset == 'airport':
            adj, features, labels = load_data_airport(dataset, data_path, return_label=True)
            val_prop, test_prop = 0.15, 0.15
        elif dataset[:3] == 'hrg':
            raise Exception("HRG with NC not implemented.")
        elif dataset == 'lfr':
            raise Exception("LFR with NC not implemented.")
        elif dataset == 'sbm':
            raise Exception("SBM with NC not implemented.")
        else:
            raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)

    labels = torch.LongTensor(labels)
    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}
    return data


# ############### DATASETS ####################################


def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + 500)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not use_feats:
        features = sp.eye(adj.shape[0])
    return adj, features, labels, idx_train, idx_val, idx_test


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_synthetic_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels


def load_data_airport(dataset_str, data_path, return_label=False):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    features = np.array([graph.node[u]['feat'] for u in graph.nodes()])
    if return_label:
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0/7, 8.0/7, 9.0/7])
        return sp.csr_matrix(adj), features, labels
    else:
        return sp.csr_matrix(adj), features


# new datasets ########################

def load_hrg_data(name, use_feats=True):
    """
    Generate and load HRG data.

    Parameters are inputted in the dataset name: for example hrg_n1000_t0.2 will generate
    a HRG with 1000 nodes and temperature 0.2.

    If parameters are not specified, defaults are n = 100 and t = 0.0.

    Classes are also generated based on splitting disk in 6.
    """

    # read parameters
    n = 1000
    t = 0.0
    alpha = 0.75
    deg = 10
    params = name.split("_")
    if len(params) > 1:
        for param in params[1:]:
            if param[0] == "n":
                n = int(param[1:])
            elif param[0] == "t":
                t = float(param[1:])
            elif param[0] == "a":
                alpha = float(param[1:])
            elif param[0] == "d":
                deg = float(param[1:])
            else:
                raise Exception(f"Invalid parameter for HRG: {param[0]}")
    
    if t == 1:
        t = 0
        random_edges = True
    else:
        random_edges = False

    # generate data
    file="data\hrg\hrg"
    edge=1
    coord=1
    rseed=12
    aseed=130
    sseed=1400
    # print(f"Generating HRG with n = {n} and t = {t}")
    subprocess.call("C:\Program Files\girgs\genhrg" + f" -n {n} -alpha {alpha} -t {t} -deg {deg} -file {file} -edge {edge} -coord {coord} -rseed {rseed} -aseed {aseed} -sseed {sseed}",
                    stdout=subprocess.DEVNULL)

    # load data
    with open("data\\hrg\\hrg.txt") as f:
        graph_file_str = f.read()
    adj = hrg_adjacency_matrix_from_str(graph_file_str)
    with open("data\\hrg\\hrg.hyp") as f:
        hyp_file_str = f.read()
    features, angles = hrg_features_array_from_str(hyp_file_str)
    if not use_feats:
        features = sp.eye(adj.shape[0])
    
    if random_edges:
        adj = nx.adjacency_matrix(nx.gnp_random_graph(n, 0.05))

    # generate labels
    num_bins = 6
    bins = []
    for i in range(1, num_bins):
        bins.append(i * 2 * np.pi / num_bins)
    labels = np.digitize(angles, bins)

    return adj, features, labels

def hrg_features_array_from_str(features_str: str):
    """
    Make array from features string in format:

    "feature feature ...
    feature feature ...
    ..."

    Saves them as cartesian not polar coordinates.
    """
    features = []
    angles = []
    for line in features_str.split("\n"):
        if len(line) == 0:
            break
        s = line.split(" ")
        r = float(s[0])
        theta = float(s[1])
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        features.append((x, y))
        angles.append(theta)
    return np.array(features), angles

def hrg_adjacency_matrix_from_str(graph_str: str):
    """
    Make nx adjacency matrix from string containing the edges, in the following format:
    
    "num_nodes num_edges

    node_id node_id
    node_id node id
    ..."

    """
    edges = []
    num_nodes = int(graph_str.split("\n")[0].split(" ")[0])
    for line in graph_str.split("\n")[2:]:
        if len(line) == 0:
            break
        s = line.split(" ")
        c1 = int(s[0])
        c2 = int(s[1])
        edges.append((c1, c2))

    # to dict
    edges_dict = collections.defaultdict(lambda: [])
    for edge in edges:
        edges_dict[edge[0]].append(edge[1])
        edges_dict[edge[1]].append(edge[0])
    
    # sort dict
    sorted_dict = {}
    for i in range(num_nodes):    
        if i in edges_dict:
            sorted_dict[i] = edges_dict[i]
        else:
            sorted_dict[i] = []

    # to nx
    return nx.adjacency_matrix(nx.from_dict_of_lists(sorted_dict))


def load_lfr_data(n=1000):
    """
    Generate an LFR benchmark graph as a dataset with set parameters and seed.

    Parameters
    ----------
    n : int, optional
        Total number of nodes, by default 2000
    """
    graph = LFR(n=n, tau1=2, tau2=1.1, mu=0.1, min_degree=20, max_degree=50, seed=10)

    adj = nx.adjacency_matrix(graph)

    features = sp.eye(adj.shape[0])

    return adj, features



def load_sbm_data(blocks=None, transitions=None, deterministic=False):
    """
    Generate LFR benchmark graph with one-hot features and community id as label.
    """

    # define defaults
    if blocks is None:
        blocks = [200, 240, 280, 320]
    if transitions is None:
        transitions = [[0.9, 0.02, 0.03, 0.04],
                        [0.02, 0.8, 0.05, 0.06],
                        [0.03, 0.05, 0.9, 0.02],
                        [0.04, 0.06, 0.02, 0.8]]
    
    # set to deterministic if given
    if deterministic:
        for i in range(len(transitions)):
            for j in range(len(transitions)):
                if i == j:
                    transitions[i][j] = 1
                else:
                    transitions[i][j] = 0
    
    graph = nx.stochastic_block_model(blocks, transitions, seed=1)


    adj = nx.adjacency_matrix(graph)

    features = sp.eye(adj.shape[0])

    partition = graph.graph["partition"]
    labels = np.zeros((len(graph), len(blocks)))
    color_map = ["r", "c", "b", "m"]
    node_colors = []

    for i, part in enumerate(partition):
        for node in part:
            labels[node, i] = 1
            node_colors.append(color_map[i])


    # draw
    # import matplotlib.pyplot as plt
    # nx.draw(graph, node_size=40, node_color=node_colors)
    # plt.savefig("results\\datasets\\sbm_colored.png")
    # raise Exception


    # use partitions as labels and features
    # features = labels

    # np.savetxt("data\\sbm\\sbm.txt", nx.to_numpy_array(graph))

    return adj, features, labels

def load_roadnet_ca():
    
    with open("data\\roadnet_ca\\roadNet-CA.txt", 'r') as f:
        content = f.read()

    edges = []
    for line in content.split("\n"):
        if len(line) == 0 or line[0] == '#':
            continue
        vs = line.split("\t")
        v1, v2 = int(vs[0]), int(vs[1])
        edges.append((v1, v2))

    graph = nx.from_edgelist(edges)
    adj = nx.adjacency_matrix(graph)
    
    features = sp.eye(len(edges))

    return adj, features

def load_enschede_road():
    """
    Enschede road dataset previously generated using osmnx, only loaded here.
    """
    with open('data\\enschede_road\\enschede_adj.txt', 'r') as file:
        text = file.read()
    adj_dict = {}
    for line in text.split("\n"):
        nodes = line.split(" ")[:-1]
        for i, node in enumerate(nodes):
            nodes[i] = int(node)
        if len(nodes) == 0:
            continue
        adj_dict[nodes[0]] = nodes[1:]
    adj = None
    graph = nx.from_dict_of_lists(adj_dict)
    adj = nx.adj_matrix(graph)

    with open('data\\enschede_road\\enschede_pos.txt', 'r') as file:
        text = file.read()
    features_list = []
    node_positions = {}
    counter = 0
    for line in text.split("\n"):
        if len(line) == 0:
            continue
        pos = line.split(" ")[1:3]
        pos = [float(pos[0]), float(pos[1])]
        features_list.append(pos)
        node_positions[counter] = pos
        counter += 1
    features = np.array(features_list)  

    # TODO remove this part and node_positions and counter
    # nx.draw_networkx(graph, pos=node_positions, with_labels=False, node_size=1)
    

    # nx.draw_networkx_nodes(graph, pos=node_positions, alpha=1, node_size=1, node_color='r')
    # nx.draw_networkx_edges(graph, pos=node_positions, alpha=1)
    
    # plt.axis('off')
    # plt.savefig("results\\datasets\\enschede_road.png", bbox_inches='tight', dpi=300)

    # print(adj.shape)

    # plt.show()

    return adj, features

if __name__=="__main__":
    # adj, features, labels = load_synthetic_data("disease_lp", True, "data\\disease_lp")
    # print(adj.shape)
    # print(features.shape)
    # adj, features, labels = load_synthetic_data("disease_nc", True, "data\\disease_nc")
    # print(adj.shape)
    # print(features.shape)

    # load_roadnet_ca()
    load_sbm_data()
    pass