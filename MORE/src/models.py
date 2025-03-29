import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import device, logger

def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

class Attention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super(Attention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = attn / abs(attn.min())
        attn = self.dropout(F.softmax(F.normalize(attn, p=1, dim=-1), dim=-1))
        output = torch.matmul(attn, v)
        return output, attn, v

class VariLengthInputLayer(nn.Module):
    def __init__(self, modal_num, num_class, input_data_dims, d_k, d_v, n_head, dropout):
        super(VariLengthInputLayer, self).__init__()
        self.n_head = n_head
        self.dims = input_data_dims
        self.d_k = d_k
        self.d_v = d_v
        self.modal_num = modal_num
        self.w_qs = nn.ModuleList([nn.Linear(dim, n_head * d_k, bias=True) for dim in self.dims])
        self.w_ks = nn.ModuleList([nn.Linear(dim, n_head * d_k, bias=True) for dim in self.dims])
        self.w_vs = nn.ModuleList([nn.Linear(dim, n_head * d_v, bias=True) for dim in self.dims])
        self.attention = Attention(temperature=d_k ** 0.5, attn_dropout=dropout)
        self.fc = nn.Linear(n_head * d_v, n_head * d_v)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(n_head * d_v, eps=1e-6)
        self.model = nn.Sequential(nn.Linear(self.modal_num * n_head * d_v, num_class))
        self.model.apply(xavier_init)
    def forward(self, input_data, mask=None):
        bs = input_data.size(0)
        modal_num = len(self.dims)
        q = torch.zeros(bs, modal_num, self.n_head * self.d_k).to(device)
        k = torch.zeros(bs, modal_num, self.n_head * self.d_k).to(device)
        v = torch.zeros(bs, modal_num, self.n_head * self.d_v).to(device)
        temp_dim = 0
        for i in range(modal_num):
            data = input_data[:, temp_dim: temp_dim + self.dims[i]]
            temp_dim += self.dims[i]
            q[:, i, :] = self.w_qs[i](data)
            k[:, i, :] = self.w_ks[i](data)
            v[:, i, :] = self.w_vs[i](data)
        q = q.view(bs, modal_num, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(bs, modal_num, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(bs, modal_num, self.n_head, self.d_v).transpose(1, 2)
        q, attn, residual = self.attention(q, k, v)
        q = q.transpose(1, 2).contiguous().view(bs, modal_num, -1)
        residual = residual.transpose(1, 2).contiguous().view(bs, modal_num, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        q = q.view(bs, -1)
        output = self.model(q)
        return output

class TransformerEncoder(nn.Module):
    def __init__(self, input_data_dims, hyperpm, num_class):
        super(TransformerEncoder, self).__init__()
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        self.d_k = hyperpm.n_hidden
        self.d_v = hyperpm.n_hidden
        self.n_head = hyperpm.n_head
        self.dropout = hyperpm.dropout
        self.modal_num = hyperpm.nmodal
        self.InputLayer = VariLengthInputLayer(self.modal_num, num_class, self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout)
    def forward(self, x):
        output = self.InputLayer(x)
        return output

class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Classifier_1, self).__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)
    def forward(self, x):
        return self.clf(x)

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, x, G):
        G = G.to(device)
        x = torch.matmul(x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = torch.matmul(G, x)
        return x

class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid[0])
        self.hgc2 = HGNN_conv(n_hid[0], n_hid[1])
    def forward(self, x, G):
        x = self.hgc1(x, G)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        x = F.leaky_relu(x, 0.25)
        return x

def init_model_dict(input_data_dims, hyperpm, num_view, num_class, dim_list, dim_he_list, dim_hc):
    model_dict = {}
    from models import HGNN, Classifier_1, TransformerEncoder  # local import to avoid circular dependency
    for i in range(num_view):
        model_dict[f"E{i+1}"] = HGNN(dim_list[i], num_class, dim_he_list, dropout=0.5)
        model_dict[f"C{i+1}"] = Classifier_1(dim_he_list[-1], num_class)
    if num_view >= 2:
        model_dict["C"] = TransformerEncoder(input_data_dims, hyperpm, num_class)
    return model_dict