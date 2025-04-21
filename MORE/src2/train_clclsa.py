import os
import copy
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report
from utils import (
    logger, set_seed, prepare_trte_data, gen_trte_adj_mat,
    cal_sample_weight, one_hot_tensor, save_model_dict, device, prepare_pathway_dict, construct_H_with_KNN,
    construct_H_with_pathways, generate_G_from_H
)
from models_clclsa import HypergraphCLCLSA
from losses import contrastive_loss
from feature_selection import select_features, save_selected_features
from pathway_utils import convert_gene_symbols_to_ensembl


def compute_best_features(data_tr_list, data_te_list, data_folder, feature_files, fs_method, view_list, labels_all, trte_idx):
    sel_dir = os.path.join(data_folder, "selected_features")
    os.makedirs(sel_dir, exist_ok=True)

    new_tr, new_te, new_names = [], [], []

    # loop per view
    for i, (Xtr, Xte, feat_file) in enumerate(zip(data_tr_list, data_te_list, feature_files)):
        view = view_list[i]
        idx_csv = os.path.join(sel_dir, f"{view}_fs_{fs_method}.csv")

        # load original feature names
        orig_names = pd.read_csv(feat_file, header=None).iloc[:,0].tolist()

        if os.path.exists(idx_csv):
            # ---- load precomputed indices ----
            df_idx = pd.read_csv(idx_csv, header=0)
            # pick 'original_index' or fallback to 2nd column
            ser = df_idx.get("original_index", df_idx.iloc[:,1])
            nums = pd.to_numeric(ser, errors="coerce").dropna().astype(int).tolist()

            logger.info(f"[{view}] loading {len(nums)} pre‑selected features from {idx_csv}")
            sel_names = [orig_names[j] for j in nums]
            Xtr_sel = Xtr[:, nums]
            Xte_sel = Xte[:, nums]

        else:
            # ---- compute & save ----
            y = labels_all[trte_idx["tr"]]
            Xtr_np, Xte_np = Xtr.cpu().numpy(), Xte.cpu().numpy()

            Xtr_sel_np, Xte_sel_np, nums, sel_names = select_features(
                Xtr_np, Xte_np, y, orig_names, method=fs_method
            )
            logger.info(f"[{view}] computed {len(nums)} features via {fs_method}")

            # save indices + names under selected_features/
            save_selected_features(orig_names, nums, data_folder, view, fs_method)

            # back to torch
            Xtr_sel = torch.FloatTensor(Xtr_sel_np).to(device)
            Xte_sel = torch.FloatTensor(Xte_sel_np).to(device)

        # append into new lists (if we loaded, still convert to torch here)
        if isinstance(Xtr_sel, np.ndarray):
            Xtr_sel = torch.FloatTensor(Xtr_sel).to(device)
            Xte_sel = torch.FloatTensor(Xte_sel).to(device)

        new_tr.append(Xtr_sel)
        new_te.append(Xte_sel)
        new_names.append(sel_names)

    # finally swap in your filtered data
    data_tr_list = new_tr
    data_te_list = new_te
    feature_list = new_names

    return data_tr_list, data_te_list, feature_list

def preprocess_features(features):
    gene_symbols = []
    symbol_to_original = {}
    
    for idx, feature in enumerate(features):
        if '|' in feature:
            symbol = feature.split('|')[0]
            gene_symbols.append(symbol)
            symbol_to_original[symbol] = {
                'original_feature': feature,
                'index': idx
            }
    
    symbol_to_ensembl = convert_gene_symbols_to_ensembl(gene_symbols)    
    ensembl_mapping = {}
    failed_features = []
    
    for symbol, original_info in symbol_to_original.items():
        if symbol in symbol_to_ensembl:
            ensembl_mapping[symbol_to_ensembl[symbol]] = original_info['index']

    logger.info(f"Total features processed: {len(features)}")
    logger.info(f"Features with '|': {len(gene_symbols)}")
    logger.info(f"Successfully converted to ENSEMBL: {len(ensembl_mapping)}")
    logger.info(f"Features without ENSEMBL ID: {len(failed_features)}")
    
    return ensembl_mapping

def subset_mrna_data(data_tr_list, data_te_list, file_path):
    features = pd.read_csv(file_path)
    # features = features[0].tolist()
    features = features['feature_name'].tolist()
    
    ensembl_mapping = preprocess_features(features)
    selected_indices = list(ensembl_mapping.values())
    selected_features = list(ensembl_mapping.keys())

    data_tr_list = data_tr_list[:, selected_indices]
    data_te_list = data_te_list[:, selected_indices]
    
    logger.info(f"View {1} data shape after ensembl subsetting: {data_tr_list.shape} (train), {data_te_list.shape} (test)")
    return data_tr_list, data_te_list, selected_features

def gen_trte_adj_mat(data_tr_list, data_te_list, data_tr_list_path, data_te_list_path, feature_names, k_neigs, pathway_dict=None):
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        logger.info(f"Constructing hypergraph incidence matrix for view {i+1}")
        if pathway_dict and i == 0 and feature_names:
            H_1 = construct_H_with_pathways(
                data_tr_list_path,
                data_tr_list[i], 
                pathway_dict, 
                feature_names,  
                k_neigs,
                "variance"
            )
            H_2 = construct_H_with_pathways(
                data_te_list_path,
                data_te_list[i], 
                pathway_dict, 
                feature_names, 
                k_neigs,
                "variance"
            )
        else:
            H_1 = construct_H_with_KNN(data_tr_list[i], k_neigs, is_probH=True, m_prob=1)
            H_2 = construct_H_with_KNN(data_te_list[i], k_neigs, is_probH=True, m_prob=1)
            
        adj_train_list.append(generate_G_from_H(H_1, variable_weight=False))
        adj_test_list.append(generate_G_from_H(H_2, variable_weight=False))
    return adj_train_list, adj_test_list


def train_test_CLCLSA(
    data_folder, view_list, num_class,
    lr=1e-3, epochs=200, hidden_dims=[400,200],
    latent_dim=128, attn_heads=4, lambda_contrast=0.25,
    test_interval=10, seed=42, fs_method=None,
):
    set_seed(seed)

    # prepare pathway database
    pathway_dict = prepare_pathway_dict(data_folder, file='./data/reactome_pathways.json')
    pathway_sizes = [len(genes) for genes in pathway_dict.values()]
    avg_genes_per_pathway = sum(pathway_sizes) / len(pathway_sizes)
    print(f"No. of pathways: {len(pathway_dict.keys())}")
    print(f"Average genes per pathway: {avg_genes_per_pathway:.2f}")
    print(f"Smallest pathway: {min(pathway_sizes)} genes")
    print(f"Largest pathway: {max(pathway_sizes)} genes")

    # Load data
    data_tr_list, data_te_list, trte_idx, labels_all = prepare_trte_data(data_folder, view_list)
    labels_tr = torch.LongTensor(labels_all[trte_idx['tr']]).to(device)
    labels_te = torch.LongTensor(labels_all[trte_idx['te']]).to(device)
    
    feature_files = [os.path.join(data_folder, f"{v}_featname.csv") for v in view_list]
    
    if fs_method:
        init_data_tr_list, init_data_te_list = data_tr_list[0], data_te_list[0]
        for i, view in enumerate(view_list):
            file_path = os.path.join(data_folder, f'{view}_fs_{fs_method}.csv')
            
            try:
                logger.info(f"View {i+1} data shape before subsetting: {data_tr_list[i].shape} (train), {data_te_list[i].shape} (test)")
                selected_features_df = pd.read_csv(file_path)
                selected_indices = selected_features_df['original_index'].astype(int).tolist()

                data_tr_list[i] = data_tr_list[i][:, selected_indices]
                data_te_list[i] = data_te_list[i][:, selected_indices]
                
                logger.info(f"View {i+1} data shape after subsetting imp features: {data_tr_list[i].shape} (train), {data_te_list[i].shape} (test)")
                print(f"Selected {len(selected_indices)} features for {view} view")
            
            except FileNotFoundError:
                print(f"No feature selection file found for {view} view. Running the feature-selection method")
                data_tr_list, data_te_list, features = compute_best_features(data_tr_list, data_te_list,
                                                                               data_folder, feature_files, fs_method, 
                                                                               view_list, labels_all, trte_idx)
            except Exception as e:
                print(f"Error processing features for {view} view: {e}")
    
    # pathway hypergraph generation
    file_path = os.path.join(data_folder, f'1_fs_{fs_method}.csv')
    print(file_path)
    data_tr_list_path, data_te_list_path, selected_features = subset_mrna_data(data_tr_list[0], data_te_list[0], file_path)

    # Build per-view adjacency lists
    # from utils import construct_H_with_KNN, generate_G_from_H
    # adj_tr_list = []
    # adj_te_list = []
    # # construct separate hypergraph and adjacency for each omics view
    # for x_tr, x_te in zip(data_tr_list, data_te_list):
    #     H_tr = construct_H_with_KNN(x_tr, k_neigs=4, is_probH=True)
    #     H_te = construct_H_with_KNN(x_te, k_neigs=4, is_probH=True)
    #     G_tr = generate_G_from_H(H_tr, variable_weight=False)
    #     G_te = generate_G_from_H(H_te, variable_weight=False)
    #     adj_tr_list.append(G_tr)
    #     adj_te_list.append(G_te)

    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_te_list, data_tr_list_path, data_te_list_path, selected_features, 4, pathway_dict)

    # Model + optimizer
    input_dims = [x.shape[1] for x in data_tr_list]
    num_views  = len(view_list)
    model      = HypergraphCLCLSA(
        input_dims, hidden_dims, latent_dim,
        num_views, num_class, attn_heads
    ).to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    criterion  = torch.nn.CrossEntropyLoss()
    best_f1, best_state = 0.0, None

    for epoch in range(1, epochs+1):
        model.train(); optimizer.zero_grad()
        z_views, z_fused, logits = model(data_tr_list, adj_tr_list)
        cls_loss  = criterion(logits, labels_tr)
        cont_loss = contrastive_loss(z_views, labels_tr)
        loss = cls_loss + lambda_contrast * cont_loss
        loss.backward(); optimizer.step()

        if epoch % test_interval == 0:
            model.eval()
            with torch.no_grad():
                _, _, logits_val = model(data_te_list, adj_te_list)
                preds = logits_val.argmax(dim=1)
                f1 = f1_score(labels_te.cpu(), preds.cpu(), average='macro')
                logger.info(f"Epoch {epoch}: Loss {loss:.4f}, Val F1 {f1:.4f}")
                if f1 > best_f1:
                    best_f1, best_state = f1, copy.deepcopy(model.state_dict())

    # Save & report
    model.load_state_dict(best_state)
    save_model_dict(os.path.join(data_folder, 'models_clclsa'), {'model': model})
    model.eval()
    with torch.no_grad():
        _, _, logits_f = model(data_te_list, adj_te_list)
        report = classification_report(labels_te.cpu(), logits_f.argmax(1).cpu())
        logger.info(f"Best Test F1: {best_f1:.4f}")
        logger.info("Classification Report:\n" + report)

        
