import pathlib
import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


PROJECT_NAME = '基于CNN-Transformer跨模态融合的穿搭推荐模型研究'
ROOT_DIR = pathlib.Path(__file__).parent.parent.parent.parent.absolute()
WANDB_KEY = 'd88f9f90e3e7f7459c00a66f323751a06e87d997'

@dataclass
class BaseTrainConfig(ABC):
    # 数据集配置
    dataset_name: str = 'polyvore'
    # 分布式配置
    n_workers_per_gpu: int = 4
    world_size: int = -1
    backend:Literal['nccl', 'gloo']='gloo'
    @property
    @abstractmethod
    def batch_sz_per_gpu(self) -> int:
        pass


    # 训练配置
    n_epochs: int = 200
    lr: float = 2e-5
    checkpoint: str = None
    accumulation_steps: int = 4
    seed: int = 42

    # 日志配置
    wandb_key: str = WANDB_KEY
    project_name: str = PROJECT_NAME
    LOG_DIR:pathlib.Path = None
    @property
    @abstractmethod
    def auto_save_checkpoint(self) -> bool:
        pass
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    # 模式配置
    demo: bool = False

    def __post_init__(self):
        self.dataset_dir = ROOT_DIR / 'datasets' / self.dataset_name
        self.checkpoint_dir = ROOT_DIR / 'checkpoints' / self.project_name
        self.precomputed_embedding_dir = self.dataset_dir / 'precomputed_embeddings'
        if self.world_size == -1:
            self.world_size = torch.cuda.device_count()

        if self.dataset_name == 'polyvore':
            self.polyvore_type:Literal['nondisjoint', 'disjoint'] ='nondisjoint'

