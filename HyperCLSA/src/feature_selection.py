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