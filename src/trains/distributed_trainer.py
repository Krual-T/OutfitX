import json
import os
from abc import ABC, abstractmethod
import logging
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset

from src.trains.configs import BaseTrainConfig


class DistributedTrainer(ABC):
    """
    Encapsulates PyTorch distributed setup, training loop, and cleanup.

    Usage:
        def build_model(): return MyModel()
        def build_dataset(): return MyDataset()
        def build_optimizer(model): return torch.optim.Adam(model.parameters(), lr=1e-3)
        def train_step(batch, model, optimizer, scaler=None):
            # implement forward, loss, backward, step
            # return loss_value

        runner = DistributedTrainer(
            build_model_fn=build_model,
            build_dataset_fn=build_dataset,
            build_optimizer_fn=build_optimizer,
            train_step_fn=train_step,
            epochs=10,
            batch_size=32,
            use_amp=True,
        )
        runner.run()
    """
    def __init__(
        self,
        cfg:BaseTrainConfig,
    ):
        self.cfg = cfg

        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        # self.num_workers = cfg.n_workers_per_gpu
        # self.batch_size = cfg.batch_sz_per_gpu

        self.model :nn.Module = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.logger = None
        self._entered = False



    @abstractmethod
    def load_model(self)->nn.Module:
        pass
    @abstractmethod
    def load_optimizer(self)->torch.optim.Optimizer:
        pass
    @abstractmethod
    def load_scheduler(self):
        pass
    @abstractmethod
    def load_scaler(self):
        pass
    @abstractmethod
    def train_step(self):
        pass
    @abstractmethod
    def load_dataset(self)->Dataset:
        """
        加载单个dataset
        :return:
        """
        pass
    @abstractmethod
    def load_sampler(self,dataset:Dataset)->DistributedSampler:
        """
        加载单个sampler
        :param dataset:
        :return:
        """
        pass
    @abstractmethod
    def setup_data_loaders(self)->DataLoader:
        """
        加载多个dataloader
        :return:
        """
        pass
    def setup_ddp_env(self):
        dist.init_process_group(backend=self.backend, init_method="env://")

    def setup_logger(self):
        self.logger = logging.getLogger(self.cfg.project_name)

    def setup(self):
        model = self.load_model()
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            model.cuda(self.local_rank)
        self.setup_ddp_env()

        self.model:DDP = DDP(self.model, device_ids=[self.local_rank])
        self.optimizer = self.load_optimizer()
        self.scheduler = self.load_scheduler()
        # amp scaler
        self.scaler = self.load_scaler()
        self.setup_data_loaders()
        self.setup_logger()


    def train_epoch(self, epoch):
        # if self.world_size > 1: loader自定义更改
        #     self.loader.sampler.set_epoch(epoch)
        for batch in self.loader:
            # user-defined training logic
            self.train_step(batch, self.model, self.optimizer, self.scaler)

    def save_checkpoint(self, epoch):

        checkpoint_dir = self.cfg.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth')
        if self.rank == 0:
            torch.save({
                'config': self.model.module.cfg.__dict__ if self.world_size > 1 else self.model.cfg.__dict__,
                'model': self.model.state_dict()
            }, checkpoint_path)
            # TODO日志管理
        return checkpoint_path

    def flash_model(self, checkpoint_path):
        map_location = f'cuda:{self.rank}' if torch.cuda.is_available() else 'cpu'
        state_dict = torch.load(checkpoint_path, map_location=map_location, weights_only=False)['model']
        self.model.load_state_dict(state_dict)

    def __enter__(self):
        self.setup()
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        dist.destroy_process_group()

    def run(self):
        assert self._entered, "DistributedTrainer must be used in a 'with' block"
        for epoch in range(self.cfg.n_epochs):
            self.train_epoch(epoch)
            dist.barrier()
            checkpoint_path = self.save_checkpoint(epoch)
            self.flash_model(checkpoint_path)
