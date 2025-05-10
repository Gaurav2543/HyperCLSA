import torch, numpy as np
from utils import Eu_dis, hyperedge_concat, generate_G_from_H, construct_H_with_KNN_from_distance

def _radius_incidence(X, epsilon):
    dis = Eu_dis(X)
    n = dis.shape[0]
    H = np.zeros((n, n))
    for i in range(n):
        nbr = np.where(dis[i] < epsilon)[1]  # indices within radius
        if len(nbr) == 0: nbr = [i]          # self‑loop
        H[nbr, i] = 1.0
    return H

def _mutual_knn_incidence(X, k):
    dis = Eu_dis(X)
    n = dis.shape[0]
    H = np.zeros((n, n))
    for i in range(n):
        knn_i = np.argsort(dis[i]).A1[:k]
        for j in knn_i:
            knn_j = np.argsort(dis[j]).A1[:k]
            if i in knn_j:          # mutual
                H[i, j] = H[j, i] = 1.0
    return H

def build_hypergraph(X, method="knn", k=4, epsilon=1.0):
    if method == "knn":
        H = construct_H_with_KNN_from_distance(Eu_dis(X), k, True, 1)
    elif method == "radius":
        H = _radius_incidence(X, epsilon)
    elif method == "mutual_knn":
        H = _mutual_knn_incidence(X, k)
    else:
        raise ValueError(f"Un‑recognised method {method}")
    return generate_G_from_H(H, variable_weight=False)