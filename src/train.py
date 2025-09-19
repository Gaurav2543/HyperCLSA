import os
import copy
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold
from utils import (
    logger, set_seed, prepare_trte_data, # gen_trte_adj_mat (removed as it's per fold now)
    cal_sample_weight, one_hot_tensor, save_model_dict, device
)
from models import HypergraphCLCLSA
from losses import contrastive_loss
# Import feature selection functions if fs_method is used
from feature_selection import select_features, save_selected_features
# Import hypergraph construction utilities
from utils import construct_H_with_KNN, generate_G_from_H

# --- top of file: add import ---------------------------------------------
from graph_utils import build_hypergraph
# -------------------------------------------------------------------------

def train_test_CLCLSA(
    data_folder, view_list, num_class,
    lr=1e-3, epochs=200,
    hidden_dims=[400,200], latent_dim=128,
    attn_heads=4, lambda_contrast=0.25,
    graph_method="knn", k_neigs=4, radius_eps=1.0,
    test_interval=10, seed=42, fs_method=None,
    fs_kwargs={},                       # NEW – pass Boruta/RFE params
    n_splits_cv=5
):
    set_seed(seed)
    
    # Load initial data (these are split according to labels_tr/te.csv, _tr.csv, _te.csv)
    # data_tr_list_initial, data_te_list_initial are lists of torch tensors on device
    # labels_all_initial is a numpy array of all labels
    data_tr_list_initial, data_te_list_initial, _, labels_all_initial = prepare_trte_data(data_folder, view_list)

    # Reconstruct the full dataset for cross-validation
    # full_X_list will be a list of torch tensors (one for each view, full data)
    full_X_list = [torch.cat((tr, te), dim=0) for tr, te in zip(data_tr_list_initial, data_te_list_initial)]
    # full_y_np is a numpy array of all labels
    full_y_np = labels_all_initial 
    # full_y is a torch tensor of all labels on device
    full_y = torch.LongTensor(full_y_np).to(device)

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=seed)

    # Lists to store metrics from each fold
    fold_accuracies = []
    fold_f1_macros = []
    fold_f1_weighteds = []
    
    # Original feature names (load once if fs_method is active)
    original_feature_names_list = []
    if fs_method:
        feature_files = [os.path.join(data_folder, f"{v}_featname.csv") for v in view_list]
        for feat_file in feature_files:
            if os.path.exists(feat_file):
                original_feature_names_list.append(pd.read_csv(feat_file, header=None).iloc[:,0].tolist())
            else: # Fallback if featname file doesn't exist
                # This case needs careful handling if fs_method is on and names are crucial
                # For now, assume if fs_method is on, feature names are available
                logger.warning(f"Feature name file {feat_file} not found. FS might be problematic if names are needed.")
                original_feature_names_list.append([f"feat_{j}" for j in range(full_X_list[len(original_feature_names_list)].shape[1])])


    # Start cross-validation loop
    for fold_idx, (train_indices, test_indices) in enumerate(skf.split(full_X_list[0].cpu().numpy(), full_y_np)):
        logger.info(f"--- Fold {fold_idx + 1}/{n_splits_cv} ---")
        set_seed(seed + fold_idx) # Ensure reproducibility per fold, but vary augmentations/initializations

        # Split data for the current fold
        X_train_fold_list = [X[train_indices].clone() for X in full_X_list]
        y_train_fold = full_y[train_indices].clone()
        X_test_fold_list = [X[test_indices].clone() for X in full_X_list]
        y_test_fold = full_y[test_indices].clone()

        current_feature_names_list = copy.deepcopy(original_feature_names_list)

        # Feature Selection for the current fold (if fs_method is specified)
        if fs_method:
            logger.info(f"Applying feature selection ({fs_method}) for Fold {fold_idx + 1}")
            X_train_fold_fs_list = []
            X_test_fold_fs_list = []
            
            # Ensure selected_features directory exists (might be redundant if save_selected_features handles it)
            # sel_dir = os.path.join(data_folder, "selected_features")
            # os.makedirs(sel_dir, exist_ok=True)

            for i, view_idx_actual in enumerate(view_list): # i is 0-based index, view_idx_actual is from view_list (e.g., 1,2,3)
                Xtr_fold_view = X_train_fold_list[i].cpu().numpy()
                Xte_fold_view = X_test_fold_list[i].cpu().numpy()
                y_train_fold_np = y_train_fold.cpu().numpy()
                
                # Check if precomputed features for this fold exist (optional, typically FS is done fresh)
                # For simplicity, we recompute FS for each fold here.
                # If you have a strategy for precomputed per-fold features, integrate it here.
                
                logger.info(f"Performing FS for view {view_idx_actual} (index {i}) in Fold {fold_idx+1}")
                Xtr_sel_np, Xte_sel_np, sel_indices, sel_names = select_features(
                    Xtr_fold_view, Xte_fold_view, y_train_fold_np, 
                    current_feature_names_list[i], method=fs_method
                )
                
                # Save selected features for this specific fold and view
                save_selected_features(
                    current_feature_names_list[i], sel_indices, 
                    data_folder, f"view{view_idx_actual}", fs_method, fold_num=fold_idx + 1
                )

                X_train_fold_fs_list.append(torch.FloatTensor(Xtr_sel_np).to(device))
                X_test_fold_fs_list.append(torch.FloatTensor(Xte_sel_np).to(device))
                current_feature_names_list[i] = sel_names # Update for this fold
            
            # Use feature-selected data for this fold
            data_tr_list_fold = X_train_fold_fs_list
            data_te_list_fold = X_test_fold_fs_list
        else:
            data_tr_list_fold = X_train_fold_list
            data_te_list_fold = X_test_fold_list

        # ---------------- build hypergraphs per fold -------------------------
        adj_tr_list_fold, adj_te_list_fold = [], []
        for x_tr_fold, x_te_fold in zip(data_tr_list_fold, data_te_list_fold):
            G_tr = build_hypergraph(
                x_tr_fold, method=graph_method, k=k_neigs, epsilon=radius_eps
            )
            G_te = build_hypergraph(
                x_te_fold, method=graph_method, k=k_neigs, epsilon=radius_eps
            )
            adj_tr_list_fold.append(G_tr)
            adj_te_list_fold.append(G_te)

        # Model + Optimizer for the current fold
        input_dims_fold = [x.shape[1] for x in data_tr_list_fold]
        num_views_fold = len(view_list) # Should be same as len(data_tr_list_fold)
        
        model_fold = HypergraphCLCLSA(
            input_dims_fold, hidden_dims, latent_dim,
            num_views_fold, num_class, attn_heads
        ).to(device)
        
        optimizer_fold = torch.optim.Adam(model_fold.parameters(), lr=lr)
        criterion_fold = torch.nn.CrossEntropyLoss() # Or use a global one
        
        fold_best_f1_macro = 0.0 # Track best F1-macro on this fold's test set during its training
        fold_best_state = None

        # Training loop for the current fold
        for epoch in range(1, epochs + 1):
            model_fold.train()
            optimizer_fold.zero_grad()
            
            z_views_tr, z_fused_tr, logits_tr = model_fold(data_tr_list_fold, adj_tr_list_fold)
            
            cls_loss = criterion_fold(logits_tr, y_train_fold)
            cont_loss = contrastive_loss(z_views_tr, y_train_fold) # Ensure contrastive_loss handles LongTensor labels
            
            loss = cls_loss + lambda_contrast * cont_loss
            loss.backward()
            optimizer_fold.step()

            if epoch % test_interval == 0 or epoch == epochs:
                model_fold.eval()
                with torch.no_grad():
                    _, _, logits_val_fold = model_fold(data_te_list_fold, adj_te_list_fold)
                    preds_val_fold = logits_val_fold.argmax(dim=1)
                    f1_val_macro = f1_score(y_test_fold.cpu(), preds_val_fold.cpu(), average='macro', zero_division=0)
                
                logger.info(f"Fold {fold_idx+1} Epoch {epoch}/{epochs}: Loss {loss.item():.4f}, Val F1-Macro {f1_val_macro:.4f}")
                
                if f1_val_macro > fold_best_f1_macro:
                    fold_best_f1_macro = f1_val_macro
                    fold_best_state = copy.deepcopy(model_fold.state_dict())
        
        # Load best state for this fold and evaluate
        if fold_best_state:
            model_fold.load_state_dict(fold_best_state)
        
        model_fold.eval()
        with torch.no_grad():
            _, _, logits_test_fold = model_fold(data_te_list_fold, adj_te_list_fold)
            preds_test_fold = logits_test_fold.argmax(dim=1)

        # Calculate metrics for this fold
        y_test_fold_cpu = y_test_fold.cpu().numpy()
        preds_test_fold_cpu = preds_test_fold.cpu().numpy()

        acc_fold = accuracy_score(y_test_fold_cpu, preds_test_fold_cpu)
        f1_macro_fold = f1_score(y_test_fold_cpu, preds_test_fold_cpu, average='macro', zero_division=0)
        f1_weighted_fold = f1_score(y_test_fold_cpu, preds_test_fold_cpu, average='weighted', zero_division=0)

        fold_accuracies.append(acc_fold)
        fold_f1_macros.append(f1_macro_fold)
        fold_f1_weighteds.append(f1_weighted_fold)

        logger.info(f"Fold {fold_idx+1} Test Results: Acc: {acc_fold:.4f}, F1-Macro: {f1_macro_fold:.4f}, F1-Weighted: {f1_weighted_fold:.4f}")
        report_fold = classification_report(y_test_fold_cpu, preds_test_fold_cpu, zero_division=0)
        logger.info(f"Fold {fold_idx+1} Classification Report:\n{report_fold}")

    # After all folds, calculate and log mean and std dev of metrics
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mean_f1_macro = np.mean(fold_f1_macros)
    std_f1_macro = np.std(fold_f1_macros)
    mean_f1_weighted = np.mean(fold_f1_weighteds)
    std_f1_weighted = np.std(fold_f1_weighteds)

    logger.info("--- Cross-Validation Summary ---")
    logger.info(f"Accuracy:     {mean_acc:.4f} +/- {std_acc:.4f}")
    logger.info(f"F1-Macro:     {mean_f1_macro:.4f} +/- {std_f1_macro:.4f}")
    logger.info(f"F1-Weighted:  {mean_f1_weighted:.4f} +/- {std_f1_weighted:.4f}")

    # ⬇︎  add this block
    return {
        "mean_acc":           mean_acc,
        "std_acc":            std_acc,
        "mean_f1_macro":      mean_f1_macro,
        "std_f1_macro":       std_f1_macro,
        "mean_f1_weighted":   mean_f1_weighted,
        "std_f1_weighted":    std_f1_weighted,
    }

    # Note: Model saving (save_model_dict) is typically not done for the "best fold" in a CV setup.
    # CV is for robust evaluation. If a final model is needed, it's usually retrained on all available training data
    # (or the full dataset if no separate holdout test set is planned beyond CV).
    # The original save_model_dict call is omitted here.
    # If you need to save the model from the "best" fold or an average model, that logic would be added here.