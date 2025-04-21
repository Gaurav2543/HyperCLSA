import os
import copy
import torch
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report
from utils import (
    logger, set_seed, prepare_trte_data, gen_trte_adj_mat,
    cal_sample_weight, one_hot_tensor, save_model_dict, device
)
from models_clclsa import HypergraphCLCLSA
from losses import contrastive_loss
from feature_selection import select_features

args = None  # Placeholder for argument parsing

def train_test_CLCLSA(
    data_folder, view_list, num_class,
    lr=1e-3, epochs=200, hidden_dims=[400,200],
    latent_dim=128, attn_heads=4, lambda_contrast=0.5,
    test_interval=10, seed=42, fs_method=None,
):
    set_seed(seed)
    # Load data
    data_tr_list, data_te_list, trte_idx, labels_all = prepare_trte_data(data_folder, view_list)
    labels_tr = torch.LongTensor(labels_all[trte_idx['tr']]).to(device)
    labels_te = torch.LongTensor(labels_all[trte_idx['te']]).to(device)
    
    feature_files = [os.path.join(data_folder, f"{v}_featname.csv") for v in view_list]
    
    import glob

    sel_dir = os.path.join(data_folder, "selected_features")
    os.makedirs(sel_dir, exist_ok=True)

    if hasattr(args, 'feature_selection_method') and args.feature_selection_method:
        method = args.feature_selection_method.lower()

        for i, view in enumerate(view_list):
            view_num = i + 1

            # 1) look for any CSV matching the naming pattern
            pattern = os.path.join(sel_dir, f"view_{view_num}_*{method}*.csv")
            candidates = glob.glob(pattern)

            if candidates:
                # load indices from the first match
                file_to_load = candidates[0]
                df = pd.read_csv(file_to_load, header=0)

                # pick the "original_index" column if present, else the 2nd column
                if 'original_index' in df.columns:
                    idx_ser = df['original_index']
                else:
                    idx_ser = df.iloc[:,1]

                # coerce & drop any stray non‑integers
                nums = pd.to_numeric(idx_ser, errors='coerce')
                if nums.isna().any():
                    logger.warning(f"[view {view_num}] dropped {nums.isna().sum()} bad rows from {os.path.basename(file_to_load)}")
                selected_indices = nums.dropna().astype(int).tolist()

                logger.info(f"[view {view_num}] loaded {len(selected_indices)} pre‑computed indices from {os.path.basename(file_to_load)}")

                # slice your arrays
                data_tr_list[i] = data_tr_list[i][:, selected_indices]
                data_te_list[i] = data_te_list[i][:, selected_indices]

            else:
                # no saved CSV found: compute & then save indices for next time
                feat_file = os.path.join(data_folder, f"{view}_featname.csv")
                feature_names = pd.read_csv(feat_file, header=None).iloc[:,0].tolist()

                filtered_tr, filtered_te, selected_indices, selected_names = select_features(
                    data_tr_list[i],
                    data_te_list[i],
                    labels_all[trte_idx["tr"]],
                    feature_names,
                    method=method,
                )

                # overwrite with filtered data
                data_tr_list[i] = filtered_tr
                data_te_list[i] = filtered_te

                # save only the indices
                out_idx = os.path.join(sel_dir, f"view_{view_num}_fs_{method}.csv")
                pd.DataFrame(selected_indices).to_csv(out_idx, index=False, header=False)
                logger.info(f"[view {view_num}] computed & saved {len(selected_indices)} indices → {os.path.basename(out_idx)}")

    # # if user asked for FS, apply it per view
    # if fs_method:
    #     from feature_selection import select_features
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
            
    #         # overwrite only if we actually performed FS
    #         data_tr_list, data_te_list = new_tr, new_te
    #         feature_names_list = new_names
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
