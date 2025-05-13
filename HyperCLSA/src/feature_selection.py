import numpy as np
import pandas as pd
import torch
import os
from sklearn.feature_selection import (RFE)
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from boruta import BorutaPy
from utils import logger
import shap


# ----------------------------
# Boruta
# ----------------------------

def boruta_selection(X, y, max_iter=50):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    selector = BorutaPy(rf, n_estimators='auto', verbose=0,
                        random_state=42, max_iter=max_iter)
    selector.fit(X_scaled, y)
    idx = np.where(selector.support_)[0]
    return X[:, idx], idx

# ----------------------------
# Recursive Feature Elimination (RFE)
# ----------------------------

def rfe_filter(X, y, k=500, step=0.1, estimator=None):
    if estimator is None:
        estimator = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000)
    selector = RFE(estimator, n_features_to_select=k, step=step)
    selector.fit(X, y)
    mask = selector.support_
    idx = np.where(mask)[0]
    return X[:, idx], idx

# ----------------------------
# SHAP Importance (SHAP)
# ----------------------------

def shap_filter(X, y, k=500, model=None):    
    if model is None:
        model = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                      random_state=42, n_jobs=-1)
    
    # Fit the model
    model.fit(X, y)
    
    # Create the explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X)
    
    print(f"Shape of shap_values: {np.shape(shap_values)}")
    print(f"Type of shap_values: {type(shap_values)}")
    
    # Since we know it's a 3D numpy array (samples, features, classes)
    # Calculate mean absolute value across samples first
    mean_abs_per_sample = np.abs(shap_values).mean(axis=0)  # This gives (features, classes)
    
    # Then calculate mean across classes to get a single importance value per feature
    total_importance = mean_abs_per_sample.mean(axis=1)  # This gives (features,)
    
    print(f"Shape of total_importance: {total_importance.shape}")
    print(f"Top 10 importance values: {total_importance[:10]}")
    
    # Get the top k feature indices
    idx = np.argsort(total_importance)[::-1][:k]
    
    print(f"Shape of idx: {idx.shape}")
    print(f"First 10 indices: {idx[:10]}")
    
    # Return the selected features and their indices
    return X[:, idx], idx

# def shap_filter(X, y, k=500, model=None):    
#     if model is None:
#         model = RandomForestClassifier(n_estimators=100, max_depth=10, 
#                                       random_state=42, n_jobs=-1)
    
#     # Fit the model
#     model.fit(X, y)
    
#     # Create the explainer
#     explainer = shap.TreeExplainer(model)
    
#     # Calculate SHAP values
#     shap_values = explainer.shap_values(X)
    
#     print(f"Shape of shap_values: {np.shape(shap_values)}")
    
#     # Calculate mean absolute SHAP value across samples for each class
#     # This gives importance per feature per class - shape (features, classes)
#     mean_abs_per_class = np.abs(shap_values).mean(axis=0)
    
#     # Get the number of classes from the SHAP values
#     num_classes = mean_abs_per_class.shape[1]
#     num_features = mean_abs_per_class.shape[0]
    
#     # Number of features to select per class
#     k_per_class = k // num_classes
#     extra = k % num_classes  # Handle any remainder
    
#     # Set to store all selected indices
#     selected_indices = set()
    
#     # For each class, select the top k_per_class features
#     for class_idx in range(num_classes):
#         # Get importance for this class
#         class_importance = mean_abs_per_class[:, class_idx]
        
#         # Adjust k for the last class to include any remainder
#         adjusted_k = k_per_class + (extra if class_idx == num_classes-1 else 0)
        
#         # Get top indices for this class
#         top_indices = np.argsort(class_importance)[::-1][:adjusted_k]
        
#         # Add to our set of selected indices
#         selected_indices.update(top_indices)
    
#     # Convert set to sorted list
#     idx = sorted(list(selected_indices))
    
#     # If we have fewer than k features, add more based on average importance
#     if len(idx) < k:
#         # Average importance across all classes
#         avg_importance = mean_abs_per_class.mean(axis=1)
        
#         # Sort by average importance
#         sorted_indices = np.argsort(avg_importance)[::-1]
        
#         # Add features not already selected until we reach k
#         additional = [i for i in sorted_indices if i not in idx][:k - len(idx)]
#         idx.extend(additional)
    
#     # If we have too many features, trim to the top k by average importance
#     elif len(idx) > k:
#         # Average importance across all classes
#         avg_importance = mean_abs_per_class.mean(axis=1)
        
#         # Sort the selected indices by their average importance
#         idx = sorted(idx, key=lambda i: avg_importance[i], reverse=True)[:k]
    
#     print(f"Selected {len(idx)} features")
#     print(f"First 10 indices: {idx[:10]}")
    
#     # Return the selected features and their indices
#     return X[:, idx], np.array(idx)

# --- keep everything you already have above ------------------------------

def load_or_run_fs(
    X_tr, X_te, y_tr, names, cache_dir, tag,
    method="boruta", **fs_kwargs
):
    """
    If <cache_dir>/<tag>.csv exists → load indices and slice.
    Else run Boruta / RFE once, save CSV, and return slices.

    Args
    ----
    tag : str            # unique file stem per (method+params+fold+view)
    fs_kwargs : dict     # forwarded to boruta_selection / rfe_filter
    """
    os.makedirs(cache_dir, exist_ok=True)
    csv_path = os.path.join(cache_dir, f"{tag}.csv")

    # ---------- already computed? ----------
    if os.path.exists(csv_path):
        sel_idx = pd.read_csv(csv_path)["original_index"].values
        logger.info(f"[Cache‑hit] Using {csv_path}")
        return X_tr[:, sel_idx], X_te[:, sel_idx], sel_idx

    # ---------- compute once ----------
    if method == "boruta":
        X_tr_sel, sel_idx = boruta_selection(X_tr, y_tr, **fs_kwargs)
    elif method == "rfe":
        X_tr_sel, sel_idx = rfe_filter(X_tr, y_tr, **fs_kwargs)
    elif method == "shap":
        X_tr_sel, sel_idx = shap_filter(X_tr, y_tr, **fs_kwargs)
    else:
        raise ValueError(f"Unknown FS method: {method}")

    X_te_sel = X_te[:, sel_idx]
    save_selected_features(names, sel_idx, cache_dir, "view", method)
    # also save bare CSV for quick reuse
    pd.DataFrame({"original_index": sel_idx}).to_csv(csv_path, index=False)
    return X_tr_sel, X_te_sel, sel_idx

def select_features(X_train, X_test, y_train, feature_names, method='boruta'):
    """
    X_train/X_test: numpy or torch arrays,
    returns filtered arrays, indices, and feature names.
    """
    # numpy conversion
    if torch.is_tensor(X_train): X_train = X_train.cpu().numpy()
    if torch.is_tensor(X_test):  X_test  = X_test.cpu().numpy()
    if torch.is_tensor(y_train):y_train = y_train.cpu().numpy()

    logger.info(f"Applying {method} feature selection")

    if method == 'boruta':
        tr, idx = boruta_selection(X_train, y_train) 
    elif method == 'rfe':
        tr, idx = rfe_filter(X_train, y_train)
    elif method == 'shap':
        tr, idx = shap_filter(X_train, y_train)
        print(tr)
        print("+"*100)
        print(idx)
    else:
        raise ValueError(f"Unknown FS method: {method}")

    te = X_test[:, idx]
    names = [feature_names[i] for i in idx]
    logger.info(f"Selected {len(idx)} features via {method}")
    return tr, te, idx, names

def save_selected_features(feature_names, selected_indices, output_dir, view_name, method, fold_num=None):
    # Ensure the "selected_features" subdirectory exists
    sel_features_subdir = os.path.join(output_dir, "selected_features")
    os.makedirs(sel_features_subdir, exist_ok=True)
    
    sel = [feature_names[i] for i in selected_indices]
    df = pd.DataFrame({"feature_name": sel, "original_index": selected_indices})
    
    filename_suffix = f"_fold{fold_num}" if fold_num is not None else ""
    path = os.path.join(sel_features_subdir, f"{view_name}_fs_{method}{filename_suffix}.csv")
    
    df.to_csv(path, index=False)
    logger.info(f"Saved features ({method}{filename_suffix if fold_num is not None else ''}) to {path}")