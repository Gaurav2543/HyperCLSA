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

class MultiHeadCrossModalAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_head, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # Use separate projections for query, key, value for cross-modal flexibility
        self.q_linear = nn.Linear(d_model, n_head * d_k)
        self.k_linear = nn.Linear(d_model, n_head * d_k)
        self.v_linear = nn.Linear(d_model, n_head * d_v)
        self.out = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6) # Added LayerNorm

    def forward(self, q, k, v, mask=None):
        """
        q, k, v are expected to be of shape [batch_size, num_views, d_model]
        """
        bs, modal_num, _ = q.size()

        # Linear projection and split into heads
        # Note: We apply linear layers to the original q, k, v inputs
        q_proj = self.q_linear(q).view(bs, modal_num, self.n_head, self.d_k).transpose(1, 2) # [bs, n_head, modal_num, d_k]
        k_proj = self.k_linear(k).view(bs, modal_num, self.n_head, self.d_k).transpose(1, 2) # [bs, n_head, modal_num, d_k]
        v_proj = self.v_linear(v).view(bs, modal_num, self.n_head, self.d_v).transpose(1, 2) # [bs, n_head, modal_num, d_v]

        # Scaled Dot-Product Attention
        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / math.sqrt(self.d_k) # [bs, n_head, modal_num, modal_num]
        if mask is not None:
            # Apply mask: mask shape should be broadcastable, e.g., [bs, 1, 1, modal_num] or [bs, 1, modal_num, modal_num]
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = self.softmax(scores)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v_proj) # [bs, n_head, modal_num, d_v]

        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(bs, modal_num, -1) # [bs, modal_num, n_head * d_v]
        output = self.out(output) # [bs, modal_num, d_model]

        # Add & Norm (Residual connection with original query)
        output = self.layer_norm(q + self.dropout(output)) # Apply dropout to the output of attention module

        return output, attn

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
                 num_views, num_classes, attn_heads=8, use_cross_attention=True): # Added use_cross_attention
        super().__init__()
        self.use_cross_attention = use_cross_attention # Store the flag
        self.encoders    = nn.ModuleList([
            HGNNEncoder(in_dim, hidden_dims) for in_dim in input_dims
        ])
        self.projections = nn.ModuleList([
            nn.Linear(hidden_dims[-1], latent_dim) for _ in range(num_views)
        ])
        # Conditionally initialize attention mechanism
        if self.use_cross_attention:
            print("Using MultiHeadCrossModalAttention")
             # Assuming d_k = d_v = latent_dim // attn_heads for simplicity
            d_k = d_v = latent_dim // attn_heads
            if latent_dim % attn_heads != 0:
                 raise ValueError("latent_dim must be divisible by attn_heads for MultiHeadCrossModalAttention")
            self.attention = MultiHeadCrossModalAttention(latent_dim, d_k, d_v, attn_heads)
        else:
            # Original self-attention
            self.attention = nn.MultiheadAttention(latent_dim, attn_heads, batch_first=True)

        self.classifier  = nn.Linear(latent_dim, num_classes)

    def forward(self, x_list, G_list):
        z_views = []
        for enc, proj, x, G in zip(self.encoders, self.projections, x_list, G_list):
            h = enc(x, G)
            z = proj(h)
            z_views.append(z)
        Z = torch.stack(z_views, dim=1)                     # [N, V, D]

        # Apply the selected attention mechanism
        if self.use_cross_attention:
            # MultiHeadCrossModalAttention expects q, k, v
            attn_out, _ = self.attention(Z, Z, Z)           # [N, V, D] - It handles cross-modal internally
        else:
            # Original self-attention expects query, key, value
            attn_out, _ = self.attention(Z, Z, Z)           # [N, V, D]

        z_fused = attn_out.mean(dim=1)                      # [N, D] - Simple averaging after attention
        logits  = self.classifier(z_fused)                  # [N, C]
        return z_views, z_fused, logits