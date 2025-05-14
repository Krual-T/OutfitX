from dataclasses import dataclass
from typing import Literal

from .base_train_config import BaseTrainConfig


@dataclass
class PrecomputeEmbeddingConfig(BaseTrainConfig):
    find_unused_parameters = True
    batch_size: int = 4096
    dataloader_workers: int = 2
    run_name: str = 'precompute_embedding'
    n_epochs: int = 1
    auto_save_checkpoint: bool = False
