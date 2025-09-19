import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from utils import gen_trte_inc_mat, load_ft, logger
from models import HGCN, TMO

def train_epoch(data_list, g_list, label, model, optimizer, scheduler, epoch, idx_tr):
    """
    Performs one training epoch.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    if len(data_list) >= 2:
        # For multi-omics (e.g., TMO)
        evidence_a, loss = model(data_list, g_list, label, epoch, idx_tr)
    else:
        ci = model(data_list[0], g_list[0])
        loss = torch.mean(criterion(ci[idx_tr], label[idx_tr]))
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()

def evaluate_model(model, data_list, g_list, labels_tensor, indices, epoch, criterion=nn.CrossEntropyLoss()):
    """
    Evaluates the model on a given set (training or test) and returns loss, accuracy, f1, predictions, and true labels.
    """
    model.eval()
    with torch.no_grad():
        if len(data_list) >= 2:
            # Multi-omics branch
            evidence_a, loss_val = model(data_list, g_list, labels_tensor, epoch, list(range(g_list[0].shape[0])))
            logits = evidence_a[indices, :]
            loss = criterion(logits, labels_tensor[indices])
        else:
            logits = model(data_list[0], g_list[0])
            loss = torch.mean(criterion(logits[indices], labels_tensor[indices]))
        prob = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(prob, axis=1)
        true = labels_tensor[indices].cpu().numpy()
        acc = accuracy_score(true, preds)
        f1 = f1_score(true, preds, average='weighted')
    return loss.item(), acc, f1, preds, true