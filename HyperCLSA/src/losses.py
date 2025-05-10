import torch
import torch.nn.functional as F
from utils import device

def contrastive_loss(z_views, labels, temperature=0.5):
    Z = torch.stack(z_views, dim=1)                     # [N, V, D]
    N, V, D = Z.size()
    z_flat = F.normalize(Z.view(N*V, D), dim=1)
    labels_flat = labels.unsqueeze(1).repeat(1, V).view(-1)
    sim = torch.matmul(z_flat, z_flat.T) / temperature
    mask = labels_flat.unsqueeze(1).eq(labels_flat.unsqueeze(0)).float().to(device)
    exp_sim = torch.exp(sim) * (1 - torch.eye(N*V, device=device))
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    loss = -(mask * log_prob).sum(dim=1) / mask.sum(dim=1)
    return loss.mean()