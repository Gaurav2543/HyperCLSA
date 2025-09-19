import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectKBest
import os
import logging
import torch
from sklearn.decomposition import PCA
from utils import logger

def variance_filter(data, threshold=0.001):
    selector = VarianceThreshold(threshold)
    filtered_data = selector.fit_transform(data)
    selected_indices = np.where(selector.get_support())[0]
    
    return filtered_data, selected_indices

def mutual_info_filter(data, labels, k=1000):
    # Convert data and labels to numpy arrays if they are torch tensors.
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    if isinstance(k, float) and k < 1.0:
        k = max(int(data.shape[1] * k), 1)  # At least 1 feature
    else:
        k = min(k, data.shape[1])  # Can't select more features than available

    selector = SelectKBest(mutual_info_classif, k=k)
    filtered_data = selector.fit_transform(data, labels)
    selected_indices = np.where(selector.get_support())[0]

    return filtered_data, selected_indices

def pathway_guided_selection(data, feature_names, pathway_dict, min_features=100):
    if not pathway_dict:
        logger.info("No pathway dictionary provided, skipping pathway-guided selection")
        return data, np.arange(data.shape[1])
    
    # Process feature names to get gene IDs
    if feature_names and '|' in feature_names[0]:
        gene_symbols = [name.split('|')[0] for name in feature_names]
    else:
        gene_symbols = [name.split('.')[0] for name in feature_names]
        
    # Collect genes in pathways
    pathway_genes = set()
    for genes in pathway_dict.values():
        pathway_genes.update(genes)
        
    # Find indices of genes in pathways
    selected_indices = []
    for i, gene in enumerate(gene_symbols):
        if gene in pathway_genes:
            selected_indices.append(i)
            
    # If not enough features selected, add high variance features
    if len(selected_indices) < min_features:
        if isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
        else:
            data_np = data
        variances = np.var(data_np, axis=0)
        # variances = np.var(data.cpu().numpy(), axis=0)
        non_selected = [i for i in range(data.shape[1]) if i not in selected_indices]
        sorted_indices = sorted(non_selected, key=lambda i: variances[i], reverse=True)
        additional = min(min_features - len(selected_indices), len(sorted_indices))
        selected_indices.extend(sorted_indices[:additional])
    
    selected_indices = np.array(selected_indices)
    filtered_data = data[:, selected_indices]
    
    return filtered_data, selected_indices

def select_features(data_tr, data_te, labels_tr, feature_names, method='combined', 
                   variance_threshold=0.001, mi_k=1000, pathway_dict=None, 
                   min_pathway_features=100, pca_components=None):
    # Step 1: Apply basic variance filtering to both datasets
    combined_data = np.vstack([data_tr.cpu().numpy(), data_te.cpu().numpy()])
    if method in ['variance', 'combined']:
        filtered_data, var_indices = variance_filter(combined_data, threshold=variance_threshold)
        filtered_tr = data_tr[:, var_indices]
        filtered_te = data_te[:, var_indices]
        selected_indices = var_indices
        filtered_names = [feature_names[i] for i in var_indices]
        logger.info(f"Variance filtering: {len(var_indices)} features selected")
    else:
        filtered_tr = data_tr
        filtered_te = data_te
        selected_indices = np.arange(data_tr.shape[1])
        filtered_names = feature_names
    
    # Step 2: Apply mutual information filtering
    if method in ['mutual_info', 'combined']:
        filtered_tr, mi_indices = mutual_info_filter(filtered_tr, labels_tr, k=mi_k)
        filtered_te = filtered_te[:, mi_indices]
        selected_indices = selected_indices[mi_indices]
        filtered_names = [filtered_names[i] for i in mi_indices]
        logger.info(f"Mutual information: {len(mi_indices)} features selected")
    
    # Step 3: Apply pathway-guided filtering
    if method in ['pathway', 'combined'] and pathway_dict:
        filtered_tr, pathway_indices = pathway_guided_selection(
            filtered_tr, filtered_names, pathway_dict, min_pathway_features
        )
        filtered_te = filtered_te[:, pathway_indices]
        selected_indices = selected_indices[pathway_indices]
        filtered_names = [filtered_names[i] for i in pathway_indices]
        logger.info(f"Pathway-guided: {len(pathway_indices)} features selected")
    
    # Optional: Apply PCA for dimensionality reduction
    if method == 'pca' and pca_components:
        pca = PCA(n_components=pca_components)
        filtered_tr = pca.fit_transform(filtered_tr)
        filtered_te = pca.transform(filtered_te)
        selected_indices = np.arange(filtered_tr.shape[1])  # PCA creates new features
        filtered_names = [f"PC{i+1}" for i in range(filtered_tr.shape[1])]
        logger.info(f"PCA: reduced to {filtered_tr.shape[1]} components")
    
    return filtered_tr, filtered_te, selected_indices, filtered_names

def save_selected_features(feature_names, selected_indices, output_dir, view_name):
    selected_features = [feature_names[i] for i in selected_indices]
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{view_name}_selected_features.csv")
    
    df = pd.DataFrame({"feature_name": selected_features, "original_index": selected_indices})
    df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(selected_features)} selected features to {output_file}")
