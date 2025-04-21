import math
import torch
import torch.nn as nn
from utils import device

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super().__init__()
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

class HGNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dims):
        super().__init__()
        self.conv1 = HGNN_conv(in_dim, hidden_dims[0])
        self.conv2 = HGNN_conv(hidden_dims[0], hidden_dims[1])
        self.act   = nn.LeakyReLU(0.25)
        self.drop  = nn.Dropout(0.5)

    def forward(self, x, G):
        h = self.conv1(x, G)
        h = self.act(h)
        h = self.drop(h)
        h = self.conv2(h, G)
        h = self.act(h)
        return h

class HypergraphCLCLSA(nn.Module):
    def __init__(self, input_dims, hidden_dims, latent_dim,
                 num_views, num_classes, attn_heads=4):
        super().__init__()
        self.encoders    = nn.ModuleList([
            HGNNEncoder(in_dim, hidden_dims) for in_dim in input_dims
        ])
        self.projections = nn.ModuleList([
            nn.Linear(hidden_dims[-1], latent_dim) for _ in range(num_views)
        ])
        self.attention   = nn.MultiheadAttention(latent_dim, attn_heads, batch_first=True)
        self.classifier  = nn.Linear(latent_dim, num_classes)

    def forward(self, x_list, G_list):
        z_views = []
        for enc, proj, x, G in zip(self.encoders, self.projections, x_list, G_list):
            h = enc(x, G)
            z = proj(h)
            z_views.append(z)
        Z = torch.stack(z_views, dim=1)                     # [N, V, D]
        attn_out, _ = self.attention(Z, Z, Z)               # [N, V, D]
        z_fused = attn_out.mean(dim=1)                      # [N, D]
        logits  = self.classifier(z_fused)                  # [N, C]
        return z_views, z_fused, logits