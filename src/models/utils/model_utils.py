
import torch
from torch import Tensor

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def mean_pooling(
        model_output: Tensor,
        attention_mask: Tensor
) -> Tensor:
    token_embeddings = model_output[0]  # First element of model_output contains the hidden states
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    summed_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    mask_sum = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    return summed_embeddings / mask_sum