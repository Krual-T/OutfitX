from dataclasses import dataclass
from typing import Literal

from .base_train_config import BaseTrainConfig


@dataclass
class PrecomputeEmbeddingConfig(BaseTrainConfig):
    find_unused_parameters = True
    batch_sz_per_gpu: int = 128
    name: str = 'precompute_embedding'
    n_epochs: int = 1
    auto_save_checkpoint: bool = False
    backend:Literal['nccl', 'gloo']='nccl'
