from dataclasses import dataclass
from typing import Literal

from .base_train_config import BaseTrainConfig


@dataclass
class CompatibilityTrainConfig(BaseTrainConfig):
    find_unused_parameters = True
    batch_sz_per_gpu: int = 512
    name: str = 'compatibility_train'
    auto_save_checkpoint: bool = True
