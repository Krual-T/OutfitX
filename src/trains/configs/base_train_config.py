import pathlib
from abc import ABC, abstractmethod

from dataclasses import dataclass
from typing import Literal

ROOT_DIR = pathlib.Path(__file__).parent.parent.parent.parent.absolute()

@dataclass
class BaseTrainConfig(ABC):
    # 数据集配置
    dataset_name: str = 'polyvore'
    # 分布式配置
    @property
    @abstractmethod
    def batch_sz_per_gpu(self) -> int:
        pass
    n_workers_per_gpu: int = 4
    world_size: int = -1

    # 训练配置
    n_epochs: int = 200
    lr: float = 2e-5
    checkpoint: str = None
    accumulation_steps: int = 4
    seed: int = 42

    # 日志配置 TODO: 申请密钥
    wandb_key: str = None
    project_name: str = None
    # 模式配置
    demo: bool = False
    def __post_init__(self):
        self.dataset_dir = ROOT_DIR / 'datasets' / self.dataset_name
        if self.dataset_name == 'polyvore':
            self.polyvore_type:Literal['nondisjoint', 'disjoint'] ='nondisjoint'
