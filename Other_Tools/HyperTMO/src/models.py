import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# -------------------------------
# Model Definitions
# -------------------------------
class HGCN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGCN_conv, self).__init__()
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        x = G.matmul(x)
        if self.bias is not None:
            x = x + self.bias
        return x

class HGCN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGCN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGCN_conv(in_ch, n_hid[0])
        self.hgc2 = HGCN_conv(n_hid[0], n_hid[1])
        self.hgc3 = HGCN_conv(n_hid[1], n_hid[2])
        self.clf = nn.Linear(n_hid[2], n_class)
        self.fc = nn.Softplus()
    def forward(self, x, G):
        x = self.hgc1(x, G)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.hgc2(x, G)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.hgc3(x, G)
        x = F.leaky_relu(x, 0.25)
        x = self.clf(x)
        x = self.fc(x)
        return x

def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)
    return (A + B)

class TMO(nn.Module):
    def __init__(self, in_ch, classes, omics, HGCN_dims, lambda_epochs=1):
        """
        :param classes: Number of classes
        :param omics: Number of omics data types
        :param HGCN_dims: Hidden dimensions for HGCN
        """
        super(TMO, self).__init__()
        self.omics = omics
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        self.HGCNs = nn.ModuleList([HGCN(in_ch[i], self.classes, HGCN_dims) for i in range(self.omics)])
    def DS_Combin(self, alpha):
        def DS_Combin_two(alpha1, alpha2):
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            p, S, F_val, u = dict(), dict(), dict(), dict()
            for o in range(2):
                S[o] = torch.sum(alpha[o], dim=1, keepdim=True)
                F_val[o] = alpha[o] - 1
                p[o] = F_val[o] / (S[o].expand(F_val[o].shape))
                u[o] = self.classes / S[o]
            pp = torch.bmm(p[0].view(-1, self.classes, 1), p[1].view(-1, 1, self.classes))
            uv1_expand = u[1].expand(p[0].shape)
            pu = torch.mul(p[0], uv1_expand)
            uv_expand = u[0].expand(p[0].shape)
            up = torch.mul(p[1], uv_expand)
            pp_sum = torch.sum(pp, dim=(1, 2))
            pp_diag = torch.diagonal(pp, dim1=-2, dim2=-1).sum(-1)
            C = pp_sum - pp_diag
            p_a = (torch.mul(p[0], p[1]) + pu + up) / ((1 - C).view(-1, 1).expand(p[0].shape))
            u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))
            S_a = self.classes / u_a
            f_a = torch.mul(p_a, S_a.expand(p_a.shape))
            alpha_a = f_a + 1
            return alpha_a
        for o in range(len(alpha) - 1):
            if o == 0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[o + 1])
        return alpha_a
    def forward(self, X, G, y, global_step, idx):
        evidence = self.infer(X, G)
        loss = 0
        alpha = dict()
        for o_num in range(len(X)):
            alpha[o_num] = evidence[o_num] + 1
            loss += ce_loss(y[idx], alpha[o_num][idx], self.classes, global_step, self.lambda_epochs)
        alpha_a = self.DS_Combin(alpha)
        evidence_a = alpha_a - 1
        loss += ce_loss(y[idx], alpha_a[idx], self.classes, global_step, self.lambda_epochs)
        loss = torch.mean(loss)
        return evidence_a, loss
    def infer(self, input, input_G):
        evidence = dict()
        for o_num in range(self.omics):
            evidence[o_num] = self.HGCNs[o_num](input[o_num], input_G[o_num])
        return evidence