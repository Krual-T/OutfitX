import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.5, reduction='mean'):
        super().__init__()
        assert gamma >= 0, (
            f"Invalid Value for arg 'gamma': '{gamma}' \n Gamma should be non-negative"
        )
        assert 0 <= alpha <= 1, (
            f"Invalid Value for arg 'alpha': '{alpha}' \n Alpha should be in range [0, 1]"
        )
        assert reduction in ['none', 'mean', 'sum'], (
            f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self, y_hat: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        ce_loss = F.binary_cross_entropy_with_logits(y_hat, y_true, reduction="none")
        y_hat = torch.sigmoid(y_hat)

        p_t = y_hat * y_true + (1 - y_hat) * (1 - y_true)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
            loss = alpha_t * loss

        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()  # DDP safe


def safe_divide(a, b, eps=1e-7):
    return a / (b + eps)
