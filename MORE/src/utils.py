import os
import math
import copy
import numpy as np
import torch
import logging

# ----------------------------
# Logging configuration
# ----------------------------
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S,%f')
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