"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys
import collections
import random

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

import data.lfr_benchmark.lfr_benchmark as lfr_benchmark


def load_data(args, datapath):
    if args.task == 'nc':
        data = load_data_nc(args.dataset, args.use_feats, datapath, args.split_seed)
    else:
        data = load_data_lp(args.dataset, args.use_feats, datapath, args.temperature)
        # data["adj_train"] = remove_edges_randomly(data["adj_train"], args.fraction_remove_edges)
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


def process(adj, features, normalize_adj, normalize_feats):
    """
    Normalize adjancency matrix and/or features if given.
    """
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
    """
    Only for airport.
    """
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


# ############### DATA SPLITS #####################################################


def mask_edges(adj, val_prop, test_prop, seed):
    """
    Split edges into train, test, and val. No shuffling.
    """
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

    # Necessary to avoid error when creating tensors
    train_edges = train_edges.astype(np.int32)
    train_edges_false = train_edges_false.astype(np.int32)
    val_edges = val_edges.astype(np.int32)
    val_edges_false = val_edges_false.astype(np.int32)
    test_edges = test_edges.astype(np.int32)
    test_edges_false = test_edges_false.astype(np.int32)
    #

    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false)  


def split_data(labels, val_prop, test_prop, seed):
    """
    Only for NC.
    """
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


def load_data_lp(dataset, use_feats, data_path, temperature=None):
    if dataset in ['cora', 'pubmed']:
        adj, features = load_citation_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'disease_lp':
        adj, features = load_synthetic_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'airport':
        adj, features = load_data_airport(dataset, data_path, return_label=False)
    elif dataset == 'hrg':
        adj, features = load_hrg_data(temperature, use_feats)[:2]
    elif dataset == 'lfr':
        adj, features = load_lfr_data()[:2]
    elif dataset == 'sbm':
        adj, features = load_sbm_data()[:2]
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
        elif dataset == 'hrg':
            raise Exception("HRG with NC not implemented.")
        elif dataset == 'lfr':
            adj, features, labels = load_lfr_data()
            val_prop, test_prop = 0.15, 0.15
        elif dataset == 'sbm':
            adj, features, labels = load_sbm_data()
            val_prop, test_prop = 0.15, 0.15
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
        # one-hot encoding of nodes
        features = sp.eye(adj.shape[0])

    return adj, features, labels, idx_train, idx_val, idx_test


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_synthetic_data(dataset_str, use_feats, data_path):
    """
    For DISEASE dataset.
    """
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
    # features contains 4 features from paper and 1 label (population of country),
    # which is then split into 4 bins
    if return_label:
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0/7, 8.0/7, 9.0/7])
        return sp.csr_matrix(adj), features, labels
    else:
        return sp.csr_matrix(adj), features

# #################### ADDED METHODS ####################################

def load_hrg_data(temperature=None, use_feats=True, data_path="data/hrg/", txt_file_name="hrg", hyp_file_name="hrg"):
    """
    Load HRG generated data, use 2d hyperbolic coordinates as features and generate label
    based on slicing the hyperbolic disk on angle, currently in 6 pieces.
    """
    if temperature is None or temperature == 0.0:
        temperature_str = ""
    else:
        temperature_str = str(temperature)[0] + str(temperature)[2:]
    txt_file_name = txt_file_name + temperature_str + ".txt"
    hyp_file_name = hyp_file_name + temperature_str + ".hyp"
    with open(os.path.join(data_path, hyp_file_name)) as f:
        hyp_file_str = f.read()
    features, angles = hrg_features_array_from_str(hyp_file_str)
    
    with open(os.path.join(data_path, txt_file_name), 'r') as f:
        graph_file_str = f.read()
    adj = hrg_adjacency_matrix_from_str(graph_file_str)

    num_bins = 6
    bins = []
    for i in range(1, num_bins):
        bins.append(i * 2 * np.pi / num_bins)
    labels = np.digitize(angles, bins)
    
    if not use_feats:
        # one-hot encoding of nodes
        features = sp.eye(adj.shape[0])

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

def load_lfr_data(n=10000, tau1=3, tau2=2, mu=0.5, average_degree=6):
    """
    Generate LFR benchmark graph with one-hot features and community id as label.
    """
    graph = lfr_benchmark.LFR_benchmark_graph(n=n, tau1=tau1, tau2=tau2, mu=mu, average_degree=average_degree)
    adj = nx.adjacency_matrix(graph)

    features = sp.eye(adj.shape[0])

    communities = nx.get_node_attributes(graph, "community")
    counter = 0
    for k in communities:
        if type(communities[k]) == set:
            for i in communities[k]:
                communities[i] = counter
            counter += 1
    labels = np.zeros([len(graph), counter])
    for i, node in enumerate(graph):
        labels[i, communities[node]] = 1

    return adj, features, labels

def load_sbm_data(blocks=[100, 120, 140, 160, 180], transitions=[[0.9, 0.02, 0.03, 0.02, 0.03],
                                                                 [0.02, 0.7, 0.1, 0.1, 0.08],
                                                                 [0.03, 0.1, 0.9, 0.02, 0.04],
                                                                 [0.02, 0.1, 0.02, 0.6, 0.26],
                                                                 [0.03, 0.08, 0.04, 0.26, 0.8]]):
    """
    Generate LFR benchmark graph with one-hot features and community id as label.
    """
    graph = nx.stochastic_block_model(blocks, transitions)
    adj = nx.adjacency_matrix(graph)

    features = sp.eye(adj.shape[0])

    partition = graph.graph["partition"]
    labels = np.zeros((len(graph), len(blocks)))

    for i, part in enumerate(partition):
        for node in part:
            labels[node, i] = 1

    return adj, features, labels

def add_random_noise(data, std, dataset_name):
    """
    Add Gaussian noise to array with given standard deviation.
    """
    if std == 0.0:
        return data
    if dataset_name == "hrg":
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

def remove_edges_randomly(adj, remove_frac):
    """
    Remove fraction of random edges from adjacency matix.

    NOT USED AS WE USE TEMPERATURE INSTEAD
    """
    G = nx.Graph(adj)
    # k = int(len(G.edges) * remove_frac)
    # to_remove=random.sample(G.edges(), k=k)
    # G.remove_edges_from(to_remove)
    return nx.adj_matrix(G)