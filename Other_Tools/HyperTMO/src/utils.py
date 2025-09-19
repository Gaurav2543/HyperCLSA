import os
import math
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# -------------------------------
# Set up logging
# -------------------------------
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger()

# -------------------------------
# Hypergraph Utility Functions
# -------------------------------
def cosine_dist(x1: torch.Tensor, x2: torch.Tensor, eps=1e-8):
    """Calculate the cosine similarity for each sample."""
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t())

def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge groups.
    """
    H = None
    for h in H_list:
        if h is not None and h != []:
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

def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    """
    Construct hypergraph incidence matrix from node distance matrix.
    """
    n_obj = dis_mat.shape[0]
    n_edge = n_obj
    H = torch.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = torch.argsort(-dis_vec)
        avg_dis = torch.mean(dis_vec)
        if not torch.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx
        index = (nearest_idx.clone().detach()).long()
        for node_idx in index[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = torch.exp(-dis_vec[node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H

def construct_H_with_KNN(X, K_neigs=[10], split_diff_scale=False, is_probH=True, m_prob=1):
    """
    Initialize multi-scale hypergraph Vertex-Edge matrix from original node feature matrix.
    """
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])
    if isinstance(K_neigs, int):
        K_neigs = [K_neigs]

    dis_mat = cosine_dist(X, X)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    return H

def load_feature_construct_H(fts, m_prob=1, K_neigs=[4], is_probH=True, split_diff_scale=False):
    """
    Construct hypergraph incidence matrix from feature data.
    """
    logger.info("Constructing hypergraph incidence matrix! (This may take several minutes...)")
    tmp = construct_H_with_KNN(fts, K_neigs=K_neigs, split_diff_scale=split_diff_scale,
                               is_probH=is_probH, m_prob=m_prob)
    H = hyperedge_concat(None, tmp)
    return H

def _generate_G_from_H(H, variable_weight=False):
    H = np.array(H)
    n_edge = H.shape[1]
    W = np.ones(n_edge)
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
        return G

def generate_G_from_H(H, variable_weight=False):
    """
    Calculate G from hypergraph incidence matrix H.
    """
    if not isinstance(H, list):
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G

def gen_trte_inc_mat(data, k_neigs):
    G_train_list = []
    for i in range(len(data)):
        H = load_feature_construct_H(data[i], K_neigs=k_neigs)
        G_train_list.append(generate_G_from_H(H))
    return G_train_list

def load_ft(omics_list, dataDir):
    """
    Load multi-omics data.
    """
    label = pd.read_csv(os.path.join(dataDir, 'labels.csv'), header=None)
    label_item = torch.LongTensor(label.values)
    cuda = True if torch.cuda.is_available() else False

    data_ft_list = []
    for omic in omics_list:
        data_ft_list.append(pd.read_csv(os.path.join(dataDir, omic + ".csv")).values)

    data_tensor_list = []
    for d in data_ft_list:
        tensor = torch.FloatTensor(d)
        if cuda:
            tensor = tensor.cuda()
        data_tensor_list.append(tensor)
    if cuda:
        label_item = label_item.cuda()
    return data_tensor_list, label_item.reshape(-1)

# -------------------------------
# Training Utility Classes/Functions
# -------------------------------
class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count