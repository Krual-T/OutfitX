from typing import Literal

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
        batch_answer: torch.Tensor,
        batch_negative_samples: torch.Tensor,
        batch_y_hat: torch.Tensors,
    ):
        query_emb = batch_y_hat
        pos_emb = batch_answer
        neg_embs = batch_negative_samples
        # 正样本距离
        pos_dist = F.pairwise_distance(query_emb, pos_emb)  # shape: (B,)

        # 所有负样本距离
        neg_dists = torch.norm(query_emb.unsqueeze(1) - neg_embs, dim=2)  # shape: (B, K)

        # L_all：对所有负样本的平均 hinge loss
        L_all = F.relu(pos_dist.unsqueeze(1) - neg_dists + self.margin).mean()

        # L_hard：只关注最“接近”的负样本
        hardest_neg = neg_dists.min(dim=1).values

        L_hard = F.relu(pos_dist - hardest_neg + self.margin).mean()

        return L_all + L_hard