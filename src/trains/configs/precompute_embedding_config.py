from dataclasses import dataclass
from base_train_config import BaseTrainConfig


@dataclass
class PrecomputeEmbeddingConfig(BaseTrainConfig):

    batch_sz_per_gpu: int = 128
    name: str = 'precompute_embedding'
    n_epochs: int = 1
    auto_save_checkpoint: bool = False
