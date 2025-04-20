import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import device

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

class MultiHeadCrossModalAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_head, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        
        self.q_linear = nn.Linear(d_model, n_head * d_k)
        self.k_linear = nn.Linear(d_model, n_head * d_k)
        self.v_linear = nn.Linear(d_model, n_head * d_v)
        self.out = nn.Linear(n_head * d_v, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        bs, modal_num, _ = q.size()
        
        # Linear projection and split into heads
        q = self.q_linear(q).view(bs, modal_num, self.n_head, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, modal_num, self.n_head, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, modal_num, self.n_head, self.d_v).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = self.softmax(scores)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(bs, modal_num, -1)
        output = self.out(output)
        return output, attn

class HierarchicalAttention(nn.Module):
    def __init__(self, modal_num, input_dim):
        super(HierarchicalAttention, self).__init__()
        self.attn_fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
    
    def forward(self, x):  # x: (bs, modal_num, dim)
        attn_scores = self.attn_fc(x)  # (bs, modal_num, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (bs, modal_num, 1)
        attended = (attn_weights * x).sum(dim=1)  # (bs, dim)
        return attended, attn_weights

class VariLengthInputLayer(nn.Module):
    def __init__(self, modal_num, num_class, input_data_dims, d_k, d_v, n_head, dropout, use_cross_modal=False):
        super(VariLengthInputLayer, self).__init__()
        self.n_head = n_head
        print(n_head)
        self.dims = input_data_dims
        self.d_k = d_k
        self.d_v = d_v
        self.modal_num = modal_num
        self.w_qs = nn.ModuleList([nn.Linear(dim, n_head * d_k, bias=True) for dim in self.dims])
        self.w_ks = nn.ModuleList([nn.Linear(dim, n_head * d_k, bias=True) for dim in self.dims])
        self.w_vs = nn.ModuleList([nn.Linear(dim, n_head * d_v, bias=True) for dim in self.dims])
        self.attention = Attention(temperature=d_k ** 0.5, attn_dropout=dropout)
        self.cross_attn = MultiHeadCrossModalAttention(d_model=n_head * d_v, d_k=d_k, d_v=d_v, n_head=n_head, dropout=dropout)
        self.fc = nn.Linear(n_head * d_v, n_head * d_v)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(n_head * d_v, eps=1e-6)
        self.hier_attn = HierarchicalAttention(modal_num, n_head * d_v)
        if use_cross_modal:
            self.model = nn.Sequential(nn.Linear(n_head * d_v, num_class))
        else:
            self.model = nn.Sequential(nn.Linear(self.modal_num * n_head * d_v, num_class))
        self.model.apply(xavier_init)
        self.use_cross_modal = use_cross_modal
        
    def forward(self, input_data, mask=None):
        bs = input_data.size(0)
        modal_num = len(self.dims)

        inputs = []
        temp_dim = 0
        for i in range(modal_num):
            data = input_data[:, temp_dim: temp_dim + self.dims[i]]
            temp_dim += self.dims[i]
            inputs.append(data)

        if self.use_cross_modal:
            inputs = torch.stack(inputs, dim=1)  # (bs, modal_num, dim_per_modal)
            projected = [self.w_vs[i](inputs[:, i, :]) for i in range(modal_num)]
            projected = torch.stack(projected, dim=1)  # (bs, modal_num, n_head * d_v)

            attn_out, _ = self.cross_attn(projected, projected, projected, mask=mask)
            out = self.dropout(self.fc(attn_out))
            out = self.layer_norm(out + projected)
            attn_pooled, attn_weights = self.hier_attn(out)  # (bs, dim), (bs, modal_num, 1)
            output = self.model(attn_pooled)

        else:
            # Project Q/K/V individually per modality
            q = torch.zeros(bs, modal_num, self.n_head * self.d_k).to(input_data.device)
            k = torch.zeros(bs, modal_num, self.n_head * self.d_k).to(input_data.device)
            v = torch.zeros(bs, modal_num, self.n_head * self.d_v).to(input_data.device)

            for i in range(modal_num):
                q[:, i, :] = self.w_qs[i](inputs[i])
                k[:, i, :] = self.w_ks[i](inputs[i])
                v[:, i, :] = self.w_vs[i](inputs[i])

            q = q.view(bs, modal_num, self.n_head, self.d_k).transpose(1, 2)
            k = k.view(bs, modal_num, self.n_head, self.d_k).transpose(1, 2)
            v = v.view(bs, modal_num, self.n_head, self.d_v).transpose(1, 2)

            q, attn, residual = self.attention(q, k, v)
            q = q.transpose(1, 2).contiguous().view(bs, modal_num, -1)
            residual = residual.transpose(1, 2).contiguous().view(bs, modal_num, -1)

            q = self.dropout(self.fc(q))
            out = self.layer_norm(q + residual)
            out = out.view(bs, -1)
            output = self.model(out)

        return output

class TransformerEncoder(nn.Module):
    def __init__(self, input_data_dims, hyperpm, num_class,cross_modal=False):
        super(TransformerEncoder, self).__init__()
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        self.d_k = hyperpm.n_hidden
        self.d_v = hyperpm.n_hidden
        self.n_head = hyperpm.n_head
        self.dropout = hyperpm.dropout
        self.modal_num = hyperpm.nmodal
        self.InputLayer = VariLengthInputLayer(self.modal_num, num_class, self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout,use_cross_modal=cross_modal)
        
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

        # Decoder to reconstruct input from embedding
        self.decoder = nn.Sequential(
            nn.Linear(n_hid[1], in_ch)
        )

    def forward(self, x, G, return_reconstruction=False, return_cluster=False):
        x1 = self.hgc1(x, G)
        x1 = F.leaky_relu(x1, 0.25)
        x1 = F.dropout(x1, self.dropout)
        x2 = self.hgc2(x1, G)
        x2 = F.leaky_relu(x2, 0.25)

        # Reconstruction branch
        reconstruction = self.decoder(x2) if return_reconstruction else None

        # Clustering branch
        if return_cluster:
            norm_x = x2 / (1e-8 + x2.norm(dim=1, keepdim=True))
            q = 1.0 / (1.0 + torch.cdist(norm_x, norm_x).pow(2))
            q = q / q.sum(dim=1, keepdim=True)
        else:
            q = None

        return x2, reconstruction, q


# class HGNN(nn.Module):
#     def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
#         super(HGNN, self).__init__()
#         self.dropout = dropout
#         self.hgc1 = HGNN_conv(in_ch, n_hid[0])
#         self.hgc2 = HGNN_conv(n_hid[0], n_hid[1])
        
#     def forward(self, x, G):
#         x = self.hgc1(x, G)
#         x = F.leaky_relu(x, 0.25)
#         x = F.dropout(x, self.dropout)
#         x = self.hgc2(x, G)
#         x = F.leaky_relu(x, 0.25)
#         return x

class PathwayGuidedAttention(nn.Module):
    def __init__(self, input_dim, pathway_dim, n_head, dropout=0.1):
        """
        Pathway-guided attention mechanism for the first modality (gene features)
        
        Args:
            input_dim: dimension of input features
            pathway_dim: dimension for pathway representation
            n_head: number of attention heads
            dropout: dropout rate
        """
        super(PathwayGuidedAttention, self).__init__()
        self.n_head = n_head
        self.pathway_dim = pathway_dim
        
        # Pathway projection layers
        self.pathway_proj = nn.Linear(input_dim, pathway_dim)
        
        # Multi-head attention for pathway guidance
        self.q_linear = nn.Linear(pathway_dim, n_head * pathway_dim)
        self.k_linear = nn.Linear(pathway_dim, n_head * pathway_dim)
        self.v_linear = nn.Linear(input_dim, n_head * pathway_dim)
        self.output_linear = nn.Linear(n_head * pathway_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x, pathway_matrix=None):
        """
        Args:
            x: input features [batch_size, feature_dim]
            pathway_matrix: adjacency matrix representing pathway connections [feature_dim, feature_dim]
        """
        batch_size = x.size(0)
        feature_dim = x.size(1)
        # Project input through pathway lens
        if pathway_matrix is not None:
            # Ensure pathway_matrix is on the right device
            pathway_matrix = pathway_matrix.to(x.device)
            if pathway_matrix.shape[0] != feature_dim:
                
                # Option 1: Simple average pooling to reduce dimensions
                reduced_size = feature_dim
                
                # Use adaptive average pooling to reduce the matrix dimensions
                # First make it a 4D tensor for 2D pooling operations
                temp_matrix = pathway_matrix.unsqueeze(0).unsqueeze(0)  # [1, 1, original_dim, original_dim]
                resized_matrix = F.adaptive_avg_pool2d(temp_matrix, (reduced_size, reduced_size))
                pathway_matrix = resized_matrix.squeeze(0).squeeze(0)  # [reduced_size, reduced_size]
            # Use pathway information to guide attention
            pathway_guided = torch.matmul(x,pathway_matrix)
        else:
            pathway_guided = x
            
        pathway_proj = self.pathway_proj(pathway_guided)
        
        # Multi-head attention
        q = self.q_linear(pathway_proj).view(batch_size, self.n_head, -1)
        k = self.k_linear(pathway_proj).view(batch_size, self.n_head, -1)
        v = self.v_linear(x).view(batch_size, self.n_head, -1)
        
        # Scaled dot-product attention
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.pathway_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        context = torch.bmm(attn, v)
        context = context.view(batch_size, -1)
        
        # Output projection
        output = self.output_linear(context)
        
        # Residual connection and layer normalization
        output = self.layer_norm(output + x)
        
        return output

class MultiModalIntegration(nn.Module):
    """
    Integrates multiple modalities using pathway-guided attention for the first modality
    and regular attention for others, then combines them using cross-modal attention.
    """
    def __init__(self, modal_dims, n_head, d_k, d_v, pathway_dim, num_class, dropout=0.1):
        super(MultiModalIntegration, self).__init__()
        self.modal_num = len(modal_dims)
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.hidden_dim = n_head * d_v
        # Pathway-guided attention for the first modality
        self.pathway_attention = PathwayGuidedAttention(modal_dims[0], pathway_dim, n_head, dropout)
        self.pathway_proj = nn.Linear(modal_dims[0], self.hidden_dim)
        # Regular attention for other modalities
        self.modal_projections = nn.ModuleList()
        for i in range(1, self.modal_num):
            self.modal_projections.append(nn.Linear(modal_dims[i], n_head * d_v))
        
        # Cross-modal attention for integration
        self.cross_attention = MultiHeadCrossModalAttention(
            d_model=n_head * d_v, 
            d_k=d_k, 
            d_v=d_v, 
            n_head=n_head, 
            dropout=dropout
        )
        
        # Hierarchical attention for modal fusion
        self.hier_attention = HierarchicalAttention(self.modal_num, n_head * d_v)
        
        # Final classifier
        self.classifier = nn.Linear(n_head * d_v, num_class)
        
    def forward(self, inputs, pathway_matrix=None):
        """
        Args:
            inputs: list of input tensors for each modality
            pathway_matrix: pathway adjacency matrix for the first modality
        """
        batch_size = inputs[0].size(0)
        
        # Apply pathway-guided attention to the first modality
        pathway_output = self.pathway_attention(inputs[0], pathway_matrix)
        
        # Project to common dimension
        modal_features = [self.pathway_proj(pathway_output)]
        # Apply regular projections to other modalities
        for i, projection in enumerate(self.modal_projections):
            modal_features.append(projection(inputs[i+1]))
        
        # Stack modalities for cross-attention
        stacked_features = torch.stack(modal_features, dim=1)  # [batch_size, modal_num, n_head * d_v]
        
        # Apply cross-modal attention
        cross_output, _ = self.cross_attention(stacked_features, stacked_features, stacked_features)
        
        # Apply hierarchical attention to fuse modalities
        fused_features, attn_weights = self.hier_attention(cross_output)
        
        # Classification
        output = self.classifier(fused_features)
        
        return output, attn_weights

class TransformerEncoderWithPathway(nn.Module):
    """
    Enhanced Transformer encoder that uses pathway-guided attention for integration
    """
    def __init__(self, input_data_dims, hyperpm, num_class, pathway_dim=64):
        super(TransformerEncoderWithPathway, self).__init__()
        self.input_data_dims = input_data_dims
        self.d_k = hyperpm.n_hidden
        self.d_v = hyperpm.n_hidden
        self.n_head = hyperpm.n_head
        self.dropout = hyperpm.dropout
        self.modal_num = hyperpm.nmodal
        
        self.integration = MultiModalIntegration(
            modal_dims=input_data_dims,
            n_head=self.n_head,
            d_k=self.d_k,
            d_v=self.d_v,
            pathway_dim=pathway_dim,
            num_class=num_class,
            dropout=self.dropout
        )
        
    def forward(self, x, pathway_matrix=None):
        # Split the input into different modalities
        inputs = []
        temp_dim = 0
        for i in range(self.modal_num):
            data = x[:, temp_dim: temp_dim + self.input_data_dims[i]]
            temp_dim += self.input_data_dims[i]
            inputs.append(data)
        
        # Apply integration with pathway guidance
        output, _ = self.integration(inputs, pathway_matrix)
        return output

def init_model_dict(input_data_dims, hyperpm, num_view, num_class, dim_list, dim_he_list, dim_hc, cross_modal=True, use_pathway_attention=False):
    model_dict = {}
    from models import HGNN, Classifier_1, TransformerEncoder, TransformerEncoderWithPathway  # local import to avoid circular dependency
    
    for i in range(num_view):
        model_dict[f"E{i+1}"] = HGNN(dim_list[i], num_class, dim_he_list, dropout=0.5)
        model_dict[f"C{i+1}"] = Classifier_1(dim_he_list[-1], num_class)
    
    if num_view >= 2:
        if use_pathway_attention:
            model_dict["C"] = TransformerEncoderWithPathway(input_data_dims, hyperpm, num_class)
        else:
            model_dict["C"] = TransformerEncoder(input_data_dims, hyperpm, num_class, cross_modal=cross_modal)
    
    return model_dict
