import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from utils import logger, one_hot_tensor, cal_sample_weight, save_model_dict, prepare_trte_data, construct_H_with_KNN, hyperedge_concat, generate_G_from_H, device

# Global variable to be set from main.py (to use command-line arguments inside train_test)
args = None

def train_epoch(num_cls, data_list, adj_list, label, one_hot_label, sample_weight, model_dict, optim_dict, train_MOSA=True):
    loss_dict = {}
    criterion = nn.CrossEntropyLoss(reduction='none')

    ###### --- ADD CONTRASTIVE LOSS SETUP --- ######
    # recon_criterion = nn.MSELoss()
    temperature = 0.5
    def contrastive_loss_fn(z, labels):
        z = F.normalize(z, dim=1)
        sim_matrix = torch.matmul(z, z.T) / temperature
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        exp_sim = torch.exp(sim_matrix) * (1 - torch.eye(labels.size(0)).to(device))
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        loss = - (mask * log_prob).sum(dim=1) / mask.sum(dim=1)
        return loss.mean()
    ################################################

    for m in model_dict:
        model_dict[m].train()
    num_view = len(data_list)

    ####### Changed to use contrastive loss ########
    for i in range(num_view):
        optim_dict[f"C{i+1}"].zero_grad()
        
        embeddings = model_dict[f"E{i+1}"](data_list[i], adj_list)
        logits = model_dict[f"C{i+1}"](embeddings)
        # Cross-entropy loss
        ce_loss = torch.mean(criterion(logits, label) * sample_weight)
        # Contrastive loss
        cont_loss = contrastive_loss_fn(embeddings, label)
        # Add decoder here for reconstruction and add MSE loss
        # recon_loss = torch.tensor(0.0).to(device)

        # Combine losses
        alpha, beta, gamma = 1.0, 0.5, 0.1
        # loss = alpha * ce_loss + beta * cont_loss + gamma * recon_loss
        loss = alpha * ce_loss + beta * cont_loss
        
        loss.backward()
        optim_dict[f"C{i+1}"].step()
        loss_dict[f"C{i+1}"] = loss.detach().cpu().item()
    ################################################
    
    ######### Original Loss Function Loop ##########
    # for i in range(num_view):
    #     optim_dict[f"C{i+1}"].zero_grad()
    #     ci = model_dict[f"C{i+1}"](model_dict[f"E{i+1}"](data_list[i], adj_list))
    #     loss = torch.mean(criterion(ci, label) * sample_weight)
    #     loss.backward()
    #     optim_dict[f"C{i+1}"].step()
    #     loss_dict[f"C{i+1}"] = loss.detach().cpu().item()
    ################################################

    if train_MOSA and num_view >= 2:
        optim_dict["C"].zero_grad()
        ci_list = [model_dict[f"E{i+1}"](data_list[i], adj_list) for i in range(num_view)]
        new_data = torch.cat(ci_list, dim=1)
        c = model_dict["C"](new_data)
        ce_loss = torch.mean(criterion(c, label) * sample_weight)
        loss = ce_loss  # Modify as above if contrastive loss on combined rep is needed
        loss.backward()
        optim_dict["C"].step()
        loss_dict["C"] = loss.detach().cpu().item()

    return loss_dict

def test_epoch(num_cls, data_list, adj_list, idx, model_dict, return_logits=False):
    for m in model_dict:
        model_dict[m].eval()
        
    num_view = len(data_list)
    ci_list = []
    
    for i in range(num_view):
        ci_list.append(model_dict[f"E{i+1}"](data_list[i], adj_list))
        
    if num_view >= 2:
        new_data = torch.cat(ci_list, dim=1)
        c = model_dict["C"](new_data)
    else:
        c = ci_list[0]
        
    if return_logits:
        return c
    else:
        prob = F.softmax(c, dim=1).detach().cpu().numpy()
        return prob

def gen_trte_adj_mat(data_tr_list, data_te_list, trte_idx, k_neigs):
    H_tr = []
    H_te = []
    for i in range(len(data_tr_list)):
        logger.info("Constructing hypergraph incidence matrix for view {}! (This may take several minutes...)".format(i+1))
        H_1 = construct_H_with_KNN(data_tr_list[i], k_neigs, split_diff_scale=False, is_probH=True, m_prob=1)
        H_tr.append(H_1)
        H_2 = construct_H_with_KNN(data_te_list[i], k_neigs, split_diff_scale=False, is_probH=True, m_prob=1)
        H_te.append(H_2)
    H_train = hyperedge_concat(H_tr[0], H_tr[1], H_tr[2])
    H_test  = hyperedge_concat(H_te[0], H_te[1], H_te[2])
    adj_train_list = generate_G_from_H(H_train, variable_weight=False)
    adj_test_list  = generate_G_from_H(H_test, variable_weight=False)
    return adj_train_list, adj_test_list

def train_test(data_folder, view_list, num_class, lr_e_pretrain, lr_e, lr_c, num_epoch_pretrain, num_epoch, k_neigs,cross_m=True):
    test_interval = 50  
    model_folder = os.path.join(data_folder, 'models')
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
        
    num_view = len(view_list)
    dim_hvcdn = pow(num_class, num_view)
    # Build hyperpm as a simple namespace with required fields for transformer:
    class HyperParams:
        pass
    hyperpm = HyperParams()
    hyperpm.dropout = 0.1
    hyperpm.n_hidden = 10
    hyperpm.n_head = 4
    hyperpm.nmodal = len(view_list)
    
    # Load data
    data_tr_list, data_te_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]]).to(device)
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = torch.FloatTensor(cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)).to(device)
    
    # Build adjacency matrices using provided k_neigs
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_te_list, trte_idx, k_neigs)
    dim_list = [x.shape[1] for x in data_tr_list]
    input_data_dim = [args.dim_he_list[-1]] * num_view  # using command-line provided hidden dims for transformer
    
    # Initialize model dictionary with provided hyperparameters
    from models import init_model_dict
    model_dict = init_model_dict(input_data_dim, hyperpm, num_view, num_class, dim_list, args.dim_he_list, dim_hvcdn,cross_modal=cross_m)
    for m in model_dict:
        model_dict[m].to(device)
    
    # Pretraining phase
    logger.info("Starting pretraining ...")
    optim_dict = {}
    for i in range(num_view):
        optim_dict[f"C{i+1}"] = torch.optim.Adam(
            list(model_dict[f"E{i+1}"].parameters()) + list(model_dict[f"C{i+1}"].parameters()),
            lr=lr_e_pretrain)
    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c)
    for epoch in range(num_epoch_pretrain):
        train_epoch(num_class, data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, train_MOSA=False)
        if epoch % 20 == 0:
            logger.info("Pretrain Epoch {} completed.".format(epoch))
    
    logger.info("Starting Training")
    # Main training phase
    optim_dict = {}
    for i in range(num_view):
        optim_dict[f"C{i+1}"] = torch.optim.Adam(
            list(model_dict[f"E{i+1}"].parameters()) + list(model_dict[f"C{i+1}"].parameters()),
            lr=lr_e)
    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c)
    criterion = nn.CrossEntropyLoss()
    best_f1 = 0.0
    best_epoch = -1
    best_model_state = None

     # Initialize metric recording lists
    train_loss_hist = []
    train_acc_hist = []
    train_f1_hist = []
    test_loss_hist = []
    test_acc_hist = []
    test_f1_hist = []
    
    for epoch in range(num_epoch + 1):
        loss_dict = train_epoch(num_class, data_tr_list, adj_tr_list, labels_tr_tensor,
                                one_hot_tensor(labels_tr_tensor, num_class), sample_weight_tr,
                                model_dict, optim_dict, train_MOSA=True)
        avg_train_loss = np.mean(list(loss_dict.values()))
        # Evaluate training and test metrics every test_interval epochs
        if epoch % test_interval == 0:
            # Training metrics
            train_logits = test_epoch(num_class, data_tr_list, adj_tr_list, trte_idx["tr"], model_dict, return_logits=True)
            train_prob = F.softmax(train_logits, dim=1)
            train_loss = criterion(train_logits, torch.LongTensor(np.array(labels_trte)[trte_idx["tr"]]).to(device)).item()
            train_pred = train_prob.argmax(1).detach().cpu().numpy()
            train_acc = accuracy_score(np.array(labels_trte)[trte_idx["tr"]], train_pred)
            train_f1 = f1_score(np.array(labels_trte)[trte_idx["tr"]], train_pred, average='macro' if num_class!=2 else 'binary')
            logger.info("Epoch {}: Train Loss {:.4f}, Train Acc {:.4f}, Train F1 {:.4f}".format(epoch, train_loss, train_acc, train_f1))
            
            # Test metrics
            test_logits = test_epoch(num_class, data_te_list, adj_te_list, trte_idx["te"], model_dict, return_logits=True)
            test_prob = F.softmax(test_logits, dim=1)
            test_loss = criterion(test_logits, torch.LongTensor(np.array(labels_trte)[trte_idx["te"]]).to(device)).item()
            test_pred = test_prob.argmax(1).detach().cpu().numpy()
            test_acc = accuracy_score(np.array(labels_trte)[trte_idx["te"]], test_pred)
            test_f1 = f1_score(np.array(labels_trte)[trte_idx["te"]], test_pred, average='macro' if num_class!=2 else 'binary')
            logger.info("Epoch {}: Test Loss {:.4f}, Test Acc {:.4f}, Test F1 {:.4f}".format(epoch, test_loss, test_acc, test_f1))

            # Append the current metrics to the lists
            train_loss_hist.append(train_loss)
            train_acc_hist.append(train_acc)
            train_f1_hist.append(train_f1)
            test_loss_hist.append(test_loss)
            test_acc_hist.append(test_acc)
            test_f1_hist.append(test_f1)
            
            if test_f1 > best_f1:
                best_f1 = test_f1
                best_epoch = epoch
                best_model_state = {k: copy.deepcopy(model_dict[k].state_dict()) for k in model_dict}

        # Combined plot for training and testing metrics (after the training loop)
    epochs = list(range(0, num_epoch + 1, test_interval))
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot Loss
    axs[0].plot(epochs, train_loss_hist, label='Train Loss')
    axs[0].plot(epochs, test_loss_hist, label='Test Loss')
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Loss Curve")
    axs[0].legend()
    
    # Plot Accuracy
    axs[1].plot(epochs, train_acc_hist, label='Train Accuracy')
    axs[1].plot(epochs, test_acc_hist, label='Test Accuracy')
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_title("Accuracy Curve")
    axs[1].legend()
    
    # Plot F1 Score
    axs[2].plot(epochs, train_f1_hist, label='Train F1 Score')
    axs[2].plot(epochs, test_f1_hist, label='Test F1 Score')
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("F1 Score")
    axs[2].set_title("F1 Score Curve")
    axs[2].legend()
    
    plt.tight_layout()
    plots_dir = f'{args.method}_{args.dataset}_plots'
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "training_metrics.png"))
    plt.close()
    
    # Save final model
    final_model_folder = os.path.join(model_folder, "final")
    if not os.path.exists(final_model_folder):
        os.makedirs(final_model_folder)
    save_model_dict(final_model_folder, model_dict)
    logger.info("Best Test F1: {:.4f} at epoch {}".format(best_f1, best_epoch))
    
    # Evaluate the best model on the test set
    for k in model_dict:
        model_dict[k].load_state_dict(best_model_state[k])
    test_logits = test_epoch(num_class, data_te_list, adj_te_list, trte_idx["te"], model_dict, return_logits=True)
    test_prob = F.softmax(test_logits, dim=1)
    y_true = np.array(labels_trte)[trte_idx["te"]]
    y_pred = test_prob.argmax(1).detach().cpu().numpy()
    
    # Classification report and confusion matrix
    report = classification_report(y_true, y_pred)
    logger.info("Classification Report for Best Model:\n" + report)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    with open(os.path.join(f'{plots_dir}/classification_report.txt'), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    logger.info("Confusion Matrix:\n" + str(cm))
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(num_class)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    # --- Add text (numbers) into each cell ---
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.savefig(os.path.join(f'{plots_dir}/confusion_matrix.png'))
    plt.close()
