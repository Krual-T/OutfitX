import torch
from torch import nn
import torch.nn.functional as F

class SetWiseRankingLoss(nn.Module):

    def __init__(
        self,
        margin:float = 2.0
    ):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        batch_y: torch.Tensor,
        batch_y_hat: torch.Tensor,
        batch_negative_samples: torch.Tensor,
        batch_negative_mask: torch.Tensor
    ):
        pos_dist  = F.pairwise_distance(batch_y_hat, batch_y)               # (B,)
        neg_dists = torch.norm(batch_y_hat.unsqueeze(1) - batch_negative_samples, dim=2)  # (B,K)
        # 有效值位置
        valid_mask = (~batch_negative_mask).float()                        # (B,K)
        valid_count = valid_mask.sum().clamp(min=1) # 标量

        # —— L_all ——
        hinge = F.relu(pos_dist.unsqueeze(1) - neg_dists + self.margin)    # (B,K)
        hinge = hinge * valid_mask
        L_all = hinge.sum() / valid_count

        # —— L_hard ——
        neg_dists = neg_dists.masked_fill(batch_negative_mask, torch.inf)  # pad→inf
        hardest = neg_dists.min(dim=1).values                              # (B,)
        L_hard = F.relu(pos_dist - hardest + self.margin).mean()

        return L_all + L_hard