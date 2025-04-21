import os
import copy
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report
from utils import (
    logger, set_seed, prepare_trte_data, gen_trte_adj_mat,
    cal_sample_weight, one_hot_tensor, save_model_dict, device
)
from models_clclsa import HypergraphCLCLSA
from losses import contrastive_loss

def train_test_CLCLSA(
    data_folder, view_list, num_class,
    lr=1e-3, epochs=200, hidden_dims=[400,200],
    latent_dim=128, attn_heads=4, lambda_contrast=0.25,
    test_interval=10, seed=42, fs_method=None,
):
    set_seed(seed)
    # Load data
    data_tr_list, data_te_list, trte_idx, labels_all = prepare_trte_data(data_folder, view_list)
    labels_tr = torch.LongTensor(labels_all[trte_idx['tr']]).to(device)
    labels_te = torch.LongTensor(labels_all[trte_idx['te']]).to(device)
    
    feature_files = [os.path.join(data_folder, f"{v}_featname.csv") for v in view_list]
    
    if fs_method:
        from feature_selection import select_features, save_selected_features

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
                # overwrite the featname CSV so downstream code sees only sel_names
                pd.DataFrame(sel_names).to_csv(feat_file, index=False, header=False)

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
        feature_names_list = new_names

    else:
        feature_names_list = feature_files

    # # if user asked for FS, apply it per view
    # if fs_method:
    #     from feature_selection import select_features, save_selected_features
    #     new_tr, new_te = [], []
    #     new_names = []
    #     for Xtr, Xte, feat_file in zip(data_tr_list, data_te_list, feature_files):
    #         names = pd.read_csv(feat_file, header=None).iloc[:,0].tolist()
    #         Xtr = Xtr.cpu().numpy()
    #         Xte = Xte.cpu().numpy()
    #         y   = labels_all[trte_idx["tr"]]

    #         Xtr_sel, Xte_sel, idx_sel, sel_names = select_features(
    #             Xtr, Xte, y, names, method=fs_method
    #         )

    #         # convert back to torch tensors and store
    #         new_tr.append(torch.FloatTensor(Xtr_sel).to(device))
    #         new_te.append(torch.FloatTensor(Xte_sel).to(device))
    #         new_names.append(sel_names)
            
    #         save_selected_features(
    #             feature_names=names,
    #             selected_indices=idx_sel,
    #             output_dir=data_folder,
    #             view_name=view_list[0],
    #             method=fs_method
    #         )
            
    #         # save the selected features
    #         pd.DataFrame(sel_names).to_csv(feat_file, "selected_features", index=False, header=False)
    #         logger.info(f"Saved selected features to {feat_file}")
            
    #     # overwrite only if we actually performed FS
    #     data_tr_list, data_te_list = new_tr, new_te
    #     feature_names_list = new_names
    # else:
    #     feature_names_list = feature_files

    # Build per-view adjacency lists
    from utils import construct_H_with_KNN, generate_G_from_H
    adj_tr_list = []
    adj_te_list = []
    # construct separate hypergraph and adjacency for each omics view
    for x_tr, x_te in zip(data_tr_list, data_te_list):
        H_tr = construct_H_with_KNN(x_tr, k_neigs=4, is_probH=True)
        H_te = construct_H_with_KNN(x_te, k_neigs=4, is_probH=True)
        G_tr = generate_G_from_H(H_tr, variable_weight=False)
        G_te = generate_G_from_H(H_te, variable_weight=False)
        adj_tr_list.append(G_tr)
        adj_te_list.append(G_te)

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
