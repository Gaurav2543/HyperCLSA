import os
import math
import copy
import numpy as np
import torch
import logging
import json
import sys
import pandas as pd
from pathway_utils import filter_pathways_by_gene_list, convert_gene_symbols_to_ensembl

# ----------------------------
# Logging configuration
# ----------------------------
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger()

# ----------------------------
# Set device and seed
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

# ----------------------------
# Data and Utility Functions
# ----------------------------
def prepare_trte_data(data_folder, view_list):
    """
    Load training and testing data along with labels.
    Expects CSV files: labels_tr.csv, labels_te.csv and for each view, "<view>_tr.csv" and "<view>_te.csv"
    """
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for v in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, f"{v}_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, f"{v}_te.csv"), delimiter=','))
    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]

    # Combine train and test for each view into one tensor
    data_tensor_list = []
    for i in range(num_view):
        data_mat = np.concatenate((data_tr_list[i], data_te_list[i]), axis=0)
        tensor = torch.FloatTensor(data_mat).to(device)
        data_tensor_list.append(tensor)

    idx_dict = {"tr": list(range(num_tr)),
                "te": list(range(num_tr, num_tr+num_te))}

    data_train_list = [data_tensor[idx_dict["tr"]].clone() for data_tensor in data_tensor_list]
    data_test_list  = [data_tensor[idx_dict["te"]].clone() for data_tensor in data_tensor_list]
    labels = np.concatenate((labels_tr, labels_te))
    return data_train_list, data_test_list, idx_dict, labels

def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim).to(device)
    y_onehot.scatter_(1, y.view(-1, 1), 1)
    return y_onehot

def cal_sample_weight(labels, num_class, use_sample_weight=True):
    if not use_sample_weight:
        return np.ones(len(labels)) / len(labels)
    count = np.zeros(num_class)
    for i in range(num_class):
        count[i] = np.sum(labels == i)
    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        sample_weight[np.where(labels == i)[0]] = count[i] / np.sum(count)
    return sample_weight

def Eu_dis(x):
    x = x.cpu().numpy()
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), axis=1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat

def hyperedge_concat(*H_list):
    H = None
    for h in H_list:
        if h is not None:
            if H is None:
                H = h
            else:
                if not isinstance(h, list):
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H

def _generate_G_from_H(H, variable_weight=False):
    H = np.array(H)
    n_edge = H.shape[1]
    W = np.ones(n_edge)  # hyperedge weights
    DV = np.sum(H * W, axis=1)
    DE = np.sum(H, axis=0)
    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T
    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        G = np.float32(G)
        return torch.tensor(G).to(device)

def generate_G_from_H(H, variable_weight=False):
    if not isinstance(H, list):
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G

def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    n_obj = dis_mat.shape[0]
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx
        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H

def construct_H_with_KNN(X, k_neigs, split_diff_scale=False, is_probH=True, m_prob=1):
    if isinstance(k_neigs, int):
        k_neigs = [k_neigs]
    dis_mat = Eu_dis(X)
    for k_neig in k_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat(H_tmp)
        else:
            H = [H_tmp]
    return H

def save_model_dict(folder, model_dict):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for module in model_dict:
        torch.save(model_dict[module].state_dict(), os.path.join(folder, module + ".pth"))

def load_model_dict(folder, model_dict):
    for module in model_dict:
        model_path = os.path.join(folder, module + ".pth")
        if os.path.exists(model_path):
            logger.info("Module {} loaded!".format(module))
            model_dict[module].load_state_dict(torch.load(model_path, map_location=device))
        else:
            logger.info("WARNING: Module {} from model_dict is not loaded!".format(module))
        model_dict[module].to(device)
    return model_dict

# prepare Pathway Functions
def prepare_pathway_dict(data_folder, file='./src/reactome_pathways.json'):
    """
    Prepares the pathway dictionary by loading from a JSON file and filtering based on gene list.
    """
    # Check if pathway file exists
    if not os.path.exists(file):
        logger.info(f"Pathway file {file} not found. Please run pathway_utils.py first.")
        return {}
    
    # Load pathway dictionary
    try:
        with open(file, 'r') as f:
            pathway_dict = json.load(f)
        logger.info(f"Loaded {len(pathway_dict)} pathways from {file}")
    except Exception as e:
        logger.info(f"Error loading pathway file: {e}")
        return {}
    
    feature_file = f'./{data_folder}/1_featname.csv'

    # Check if feature file exists
    if not os.path.exists(feature_file):
        logger.info(f"Feature file {feature_file} not found.")
        return pathway_dict
    
    # Read ENSEMBL IDs from feature file
    try:
        feature_df = pd.read_csv(feature_file, header=None)
        if (data_folder == "BRCA"):
            #preprocess GeneSymbol|GeneID into list of GeneSymbols
            features = feature_df[0].tolist()
            gene_symbols = [item.split('|')[0] for item in features]
            symbols_to_ensembl = convert_gene_symbols_to_ensembl(gene_symbols)
            ensembl_ids = list(symbols_to_ensembl.values())
        else:
            ensembl_ids_with_version = feature_df[0].tolist()
            ensembl_ids = [eid.split('.')[0] for eid in ensembl_ids_with_version]
        
        logger.info(f"Loaded {len(ensembl_ids)} ENSEMBL IDs from {feature_file}")
        
        filtered_dict = filter_pathways_by_gene_list(pathway_dict, ensembl_ids)
        
        pathway_ids = list(filtered_dict.keys())
        if pathway_ids:
            example_pathway_id = pathway_ids[0]
            example_genes = filtered_dict[example_pathway_id][:5]  # Show first 5 genes
            logger.info(f"Example pathway: {example_pathway_id} with genes: {example_genes}")
        
        return filtered_dict
        
    except Exception as e:
        logger.info(f"Error processing feature file: {e}")
        return pathway_dict

def construct_H_with_pathways(X, pathway_dict, feature_names, k_neigs=5):
    """
    Construct hypergraph incidence matrix using both pathway information and k-nearest neighbors
    
    Args:
    - X: Input feature tensor
    - pathway_dict: Dictionary of pathways with gene ENSEMBL IDs
    - feature_names: List of feature names corresponding to the input tensor
    - k_neigs: Number of neighbors to connect in traditional k-NN approach
    
    Returns:
    - Hypergraph incidence matrix
    """
    n_obj = X.shape[0]
    dis_mat = Eu_dis(X)
    
    # Create a traditional k-NN hypergraph
    H_knn = construct_H_with_KNN(X, k_neigs, is_probH=True)
    
    # Initialize pathway-based hypergraph
    H_pathway = np.zeros((n_obj, n_obj))
    
    # Preprocess feature names to extract gene identifiers
    if feature_names and '|' in feature_names[0]:
        # BRCA-like dataset with "GeneSymbol|GeneID" format
        gene_symbols = [name.split('|')[0] for name in feature_names]
        
        # Convert gene symbols to ENSEMBL IDs
        from pathway_utils import convert_gene_symbols_to_ensembl
        symbols_to_ensembl = convert_gene_symbols_to_ensembl(gene_symbols)
        gene_ids = list(symbols_to_ensembl.values())
    else:
        # Other datasets with direct ENSEMBL IDs
        gene_ids = [eid.split('.')[0] for eid in feature_names]
    
    # Create pathway-based connections
    for pathway, pathway_genes in pathway_dict.items():
        # Find indices of genes in the current feature set
        gene_indices = [i for i, gene_id in enumerate(gene_ids) if gene_id in pathway_genes]
        
        # If multiple genes from the same pathway are present
        if len(gene_indices) > 1:
            for i in range(len(gene_indices)):
                for j in range(i+1, len(gene_indices)):
                    idx1, idx2 = gene_indices[i], gene_indices[j]
                    # Use inverse distance as connection weight
                    dist = dis_mat[idx1, idx2]
                    H_pathway[idx1, idx2] = np.exp(-dist)
                    H_pathway[idx2, idx1] = np.exp(-dist)
    
    # Combine k-NN and pathway hypergraphs
    H_combined = H_knn + H_pathway
    return H_combined