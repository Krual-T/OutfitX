from dataclasses import dataclass
from typing import Literal

from src.trains.configs import BaseTrainConfig

@dataclass
class FillInTheBlankTrainConfig(BaseTrainConfig):
    find_unused_parameters = True
    batch_size: int = 1024*3
    dataloader_workers: int = 8
    polyvore_type: Literal['nondisjoint', 'disjoint'] = 'nondisjoint'
    run_name: str = 'fill_in_the_blank'
    auto_save_checkpoint: bool = True
    load_image: bool = False
    learning_rate: float = 4e-5 # learning rate
    n_epochs: int = 200
    accumulation_steps: int = 4

