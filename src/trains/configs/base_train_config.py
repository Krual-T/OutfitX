import pathlib
from pathlib import Path

import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional
from src.project_settings.info import PROJECT_NAME, PROJECT_DIR as ROOT_DIR

WANDB_KEY = 'd88f9f90e3e7f7459c00a66f323751a06e87d997'

@dataclass
class BaseTrainConfig(ABC):
    # 数据集配置
    dataset_name: str = 'polyvore'
    # 分布式配置
    dataloader_workers: int = 4
    world_size: int = -1
    backend:Literal['nccl', 'gloo']='nccl' if torch.cuda.is_available() else 'gloo'
    @property
    @abstractmethod
    def batch_size(self) -> int:
        pass
    use_amp: bool = True
    # 训练配置
    n_epochs: int = 200
    learning_rate: float = 2e-5
    checkpoint: str = None
    accumulation_steps: int = 1
    seed: int = 42
    @property
    @abstractmethod
    def find_unused_parameters(self)->bool:
        pass

    # 日志配置
    wandb_key: str = WANDB_KEY
    project_name: str = PROJECT_NAME
    LOG_DIR:Optional[Path] = ROOT_DIR / 'logs'
    @property
    @abstractmethod
    def auto_save_checkpoint(self) -> bool:
        pass
    @property
    @abstractmethod
    def run_name(self) -> str:
        pass
    # 模式配置
    demo: bool = False

    def __post_init__(self):
        self.dataset_dir:Path = ROOT_DIR / 'datasets' / self.dataset_name
        self.checkpoint_dir:Path = ROOT_DIR / 'checkpoints' / self.run_name
        self.precomputed_embedding_dir:Path = self.dataset_dir / 'precomputed_embeddings'
        if self.world_size == -1:
            self.world_size = torch.cuda.device_count()
        if self.dataset_name == 'polyvore':
            self.polyvore_type:Literal['nondisjoint', 'disjoint'] ='nondisjoint'
            self.checkpoint_dir:Path = ROOT_DIR / 'checkpoints' / self.polyvore_type /self.run_name
