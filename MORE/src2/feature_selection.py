import numpy as np
import pandas as pd
import torch
import os
from sklearn.feature_selection import (VarianceThreshold, SelectKBest, mutual_info_classif,
                                       RFE, RFECV)
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
# Lasso (L1)
# ----------------------------

def lasso_selection(X, y, cv_splits=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegressionCV(Cs=10, penalty='l1', solver='liblinear',
                               cv=StratifiedKFold(cv_splits),
                               max_iter=1000)
    clf.fit(X_scaled, y)
    coef = np.abs(clf.coef_).sum(axis=0)
    idx = np.where(coef > 1e-5)[0]
    return X[:, idx], idx

# ----------------------------
# Recursive Feature Elimination (RFE)
# ----------------------------

def rfe_filter(X, y, k=1000, step=0.1, estimator=None):
    if estimator is None:
        estimator = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000)
    selector = RFE(estimator, n_features_to_select=k, step=step)
    selector.fit(X, y)
    mask = selector.support_
    idx = np.where(mask)[0]
    return X[:, idx], idx

# ----------------------------
# mRMR
# ----------------------------

def mrmr_filter(X, y, k=1000):
    n_feat = X.shape[1]
    relevance = mutual_info_classif(X, y)
    selected = []
    remaining = set(range(n_feat))
    first = int(np.argmax(relevance))
    selected.append(first)
    remaining.remove(first)

    for _ in range(1, min(k, n_feat)):
        scores = {}
        for f in remaining:
            rel = relevance[f]
            red = np.mean([np.corrcoef(X[:, f], X[:, s])[0,1] for s in selected])
            scores[f] = rel - red
        nxt = max(scores, key=scores.get)
        selected.append(nxt)
        remaining.remove(nxt)

    idx = np.array(selected)
    return X[:, idx], idx


def select_features(X_train, X_test, y_train, feature_names, method='rfecv'):
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
    elif method == 'lasso':
        tr, idx = lasso_selection(X_train, y_train)
    elif method == 'rfe':
        tr, idx = rfe_filter(X_train, y_train)
    elif method == 'mrmr':
        tr, idx = mrmr_filter(X_train, y_train)
    else:
        raise ValueError(f"Unknown FS method: {method}")

    te = X_test[:, idx]
    names = [feature_names[i] for i in idx]
    logger.info(f"Selected {len(idx)} features via {method}")
    return tr, te, idx, names

def save_selected_features(feature_names, selected_indices, output_dir, view_name, method):
    os.makedirs(output_dir, exist_ok=True)
    sel = [feature_names[i] for i in selected_indices]
    df = pd.DataFrame({"feature_name": sel, "original_index": selected_indices})
    path = os.path.join(output_dir, f"{view_name}_fs_{method}.csv")
    df.to_csv(path, index=False)
    logger.info(f"Saved features ({method}) to {path}")
