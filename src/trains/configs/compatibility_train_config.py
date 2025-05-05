from dataclasses import dataclass
from typing import Literal

from .base_train_config import BaseTrainConfig


@dataclass
class CompatibilityTrainConfig(BaseTrainConfig):
    find_unused_parameters = True
    batch_size: int = 512
    dataloader_workers: int = 4
    polyvore_type: Literal['nondisjoint', 'disjoint'] = 'nondisjoint'
    name: str = 'compatibility_train'
    auto_save_checkpoint: bool = True
    load_image: bool = False
    learning_rate: float = 2e-5 # learning rate
    n_epochs: int = 200
    accumulation_steps: int = 4