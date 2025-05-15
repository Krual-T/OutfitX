from typing import Literal

from src.trains.configs import BaseTrainConfig


class ComplementaryItemRetrievalTrainConfig(BaseTrainConfig):
    find_unused_parameters = True
    batch_size: int = 4096
    dataloader_workers: int = 4
    polyvore_type: Literal['nondisjoint', 'disjoint'] = 'nondisjoint'
    run_name: str = 'complementary_item_retrieval'
    auto_save_checkpoint: bool = True
    load_image: bool = False
    learning_rate: float = 4e-5 # learning rate
    n_epochs: int = 200
    accumulation_steps: int = 8
    margin: float = 2.0

