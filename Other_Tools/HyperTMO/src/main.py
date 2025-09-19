import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score

from utils import load_ft, gen_trte_inc_mat, logger
from models import TMO, HGCN
from train import train_epoch, evaluate_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True,
                        help='more or hypertmo')
    parser.add_argument('--file_dir', '-fd', type=str, required=True,
                        help='The dataset file folder.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='The dataset file folder.')
    parser.add_argument('--seed', '-s', type=int, default=20,
                        help='Random seed, default=20.')
    parser.add_argument('--num_epoch', '-ne', type=int, default=10000, 
                        help='Training epochs, default: 10000.')
    parser.add_argument('--lr_e', '-lr', type=float, default=0.001,
                        help='Learning rate, default: 0.001.')
    parser.add_argument('--dim_he_list', '-dh', nargs='+', type=int, default=[400, 200, 200],
                        help='Hidden layer dimensions of HGCN.')
    parser.add_argument('--num_class', '-nc', type=int, required=True,
                        help='Number of classes.')
    parser.add_argument('--k_neigs', '-kn', type=int, default=4,
                        help='Number of vertices in hyperedge.')
    args = parser.parse_args()

    # For reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    omics_list = ['miRNA', 'meth', 'mRNA']
    eval_interval = 50  # 
    cuda = torch.cuda.is_available()
    idx_dict = {}

    # Load features and labels
    data_tensor_list, labels_tensor = load_ft(omics_list, args.file_dir)
    num_omics = len(data_tensor_list)

    # Prepare 5-fold cross validation
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    acc_res, F1_res, AUC_res = [], [], []
    fold = 0
    
    plots_dir = f'{args.method}_{args.dataset}_plots'
    os.makedirs(plots_dir, exist_ok=True)

    for idx_train, idx_test in skf.split(pd.DataFrame(data=data_tensor_list[0].cpu()),
                                          pd.DataFrame(labels_tensor.cpu())):
        logging.info(f"Starting fold {fold + 1}")
        # Prepare hypergraph incidence matrices
        g_list = []
        g = gen_trte_inc_mat(data_tensor_list, args.k_neigs)
        for i in range(len(data_tensor_list)):
            tensor_g = torch.Tensor(g[i])
            if cuda:
                tensor_g = tensor_g.cuda()
            g_list.append(tensor_g)
        idx_dict["tr"] = idx_train
        idx_dict["te"] = idx_test

        # Initialize model: use TMO for multi-omics, else HGCN
        if num_omics >= 2:
            model = TMO([x.shape[1] for x in data_tensor_list], args.num_class, num_omics, args.dim_he_list)
        else:
            model = HGCN(data_tensor_list[0].shape[1], args.num_class, args.dim_he_list)
        if cuda:
            model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_e, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.9)
        criterion = torch.nn.CrossEntropyLoss()

        # Lists to record metrics per evaluation interval
        train_loss_hist, train_acc_hist, train_f1_hist = [], [], []
        test_loss_hist, test_acc_hist, test_f1_hist = [], [], []

        best_acc = 0.0
        best_epoch = 0
        best_preds = None
        best_true = None

        # Training loop
        for epoch in range(args.num_epoch + 1):
            tr_loss = train_epoch(data_tensor_list, g_list, labels_tensor, model, optimizer, scheduler, epoch, idx_dict["tr"])
            if epoch % eval_interval == 0:
                train_loss, train_acc, train_f1, _, _ = evaluate_model(model, data_tensor_list, g_list, labels_tensor, idx_dict["tr"], epoch, criterion)
                test_loss, test_acc, test_f1, preds, true = evaluate_model(model, data_tensor_list, g_list, labels_tensor, idx_dict["te"], epoch, criterion)

                train_loss_hist.append(train_loss)
                train_acc_hist.append(train_acc)
                train_f1_hist.append(train_f1)
                test_loss_hist.append(test_loss)
                test_acc_hist.append(test_acc)
                test_f1_hist.append(test_f1)

                logging.info(f"Fold {fold+1} Epoch {epoch}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, Train F1 {train_f1:.4f}")
                logging.info(f"Fold {fold+1} Epoch {epoch}: Test Loss {test_loss:.4f}, Test Acc {test_acc:.4f}, Test F1 {test_f1:.4f}")

                if test_acc > best_acc:
                    best_acc = test_acc
                    best_epoch = epoch
                    best_preds = preds
                    best_true = true

        logging.info(f"Fold {fold+1}: Best Test Accuracy = {best_acc:.4f} at epoch {best_epoch}")

        # Plot and save training curves for this fold
        epochs = list(range(0, args.num_epoch + 1, eval_interval))
        fig, axs = plt.subplots(3, 1, figsize=(8, 12))

        axs[0].plot(epochs, train_loss_hist, label='Train Loss')
        axs[0].plot(epochs, test_loss_hist, label='Test Loss')
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].legend()
        axs[0].set_title(f"Fold {fold+1} Loss Curve")

        axs[1].plot(epochs, train_acc_hist, label='Train Accuracy')
        axs[1].plot(epochs, test_acc_hist, label='Test Accuracy')
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Accuracy")
        axs[1].legend()
        axs[1].set_title(f"Fold {fold+1} Accuracy Curve")

        axs[2].plot(epochs, train_f1_hist, label='Train F1')
        axs[2].plot(epochs, test_f1_hist, label='Test F1')
        axs[2].set_xlabel("Epoch")
        axs[2].set_ylabel("F1 Score")
        axs[2].legend()
        axs[2].set_title(f"Fold {fold+1} F1 Curve")

        plt.tight_layout()
        plt.savefig(f"{plots_dir}/fold_{fold+1}_metrics.png")
        plt.close()

        # Print classification report for best model on this fold and save confusion matrix plot
        report = classification_report(best_true, best_preds, digits=4)
        logging.info(f"Fold {fold+1} Classification Report:\n{report}")
        print(f"Fold {fold+1} Classification Report:\n{report}")

        cm = confusion_matrix(best_true, best_preds)
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"Fold {fold+1} Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(args.num_class)
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # Add text annotations (numbers) in each cell so counts are visible
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/fold_{fold+1}_confusion_matrix.png")
        plt.close()

        # Collect fold performance
        acc_res.append(best_acc)
        F1_res.append(f1_score(best_true, best_preds, average='weighted'))
        if args.num_class == 2:
            auc_val = roc_auc_score(best_true, F.softmax(torch.Tensor(best_preds), dim=0).numpy())
            AUC_res.append(auc_val)
        else:
            AUC_res.append(f1_score(best_true, best_preds, average='macro'))
        fold += 1

    logging.info('5-fold performance: '
                 f'Acc ({np.mean(acc_res):.4f} ± {np.std(acc_res):.4f})  '
                 f'F1 ({np.mean(F1_res):.4f} ± {np.std(F1_res):.4f})  '
                 f'AUC/F1_mac ({np.mean(AUC_res):.4f} ± {np.std(AUC_res):.4f})')
    print('5-fold performance: '
          f'Acc ({np.mean(acc_res):.4f} ± {np.std(acc_res):.4f})  '
          f'F1 ({np.mean(F1_res):.4f} ± {np.std(F1_res):.4f})  '
          f'AUC/F1_mac ({np.mean(AUC_res):.4f} ± {np.std(AUC_res):.4f})')
    logging.info("Finished training and evaluation.")

if __name__ == '__main__':
    main()