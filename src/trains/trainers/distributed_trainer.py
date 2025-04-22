import os
from abc import ABC, abstractmethod
import logging
import random

import numpy as np
import torch
import torch.distributed as dist

from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from tqdm import tqdm

from src.trains.configs import BaseTrainConfig


class DistributedTrainer(ABC):
    """
    must be used in a 'with' block
    and run command: torchrun --nproc_per_node=4 --master_port=12345 main.py
    with DistributedTrainer(cfg) as trainer:
        trainer.run()
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
        self.train_loader:DataLoader = None
        self.valid_loader:DataLoader = None
        self.test_loader:DataLoader = None

    def run(self):
        assert self._entered, "DistributedTrainer must be used in a 'with' block"
        for epoch in range(self.cfg.n_epochs):
            logs = []
            if self.train_loader is not None:
                self.model.train()
                if hasattr(self.train_loader.sampler, 'set_epoch'):
                    self.train_loader.sampler.set_epoch(epoch)
                logs.append(self.train_epoch(epoch))
            if self.valid_loader is not None:
                self.model.eval()
                logs.append(self.valid_epoch())
            if self.test_loader is not None:
                self.model.eval()
                logs.append(self.test_epoch())

            self.custom_task(epoch=epoch)

            dist.barrier()
            if self.train_loader is not None:
                checkpoint_path = self.save_checkpoint(epoch)
                self.load_checkpoint(checkpoint_path)

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
    def setup_dataloaders(self):
        """
        加载多个dataloader（先加载dataset，然后加载sampler，然后加载dataloader）
        :return:
        """
        pass
    @abstractmethod
    def loss(self):
        pass
    def setup_ddp_env(self):
        dist.init_process_group(backend=self.cfg.backend, init_method="env://")

    def setup_logger(self):
        self.logger = logging.getLogger(self.cfg.project_name)
        logging.basicConfig(level=logging.INFO if self.rank == 0 else logging.WARN)

    def setup_seed(self):
        random.seed(self.cfg.seed)
        os.environ['PYTHONHASHSEED'] = str(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setup(self):
        self.setup_seed()

        self.setup_ddp_env()

        model = self.load_model()
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            model.cuda(self.local_rank)
            self.model: DDP = DDP(model, device_ids=[self.local_rank])
        else:
            self.model: DDP = DDP(model)


        self.optimizer = self.load_optimizer()
        self.scheduler = self.load_scheduler()
        self.scaler = self.load_scaler()

        self.setup_data_loaders()
        self.setup_logger()

    def save_checkpoint(self, epoch):

        checkpoint_dir = self.cfg.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth')
        if self.rank == 0:
            torch.save({
                'epoch': epoch,
                'config': self.model.module.cfg.__dict__,
                'model': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict() if self.scheduler else None,
                'scaler': self.scaler.state_dict() if self.scaler else None,
            }, checkpoint_path)
            self.logger.info(f"[Checkpoint] Saved to {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, ckpt_path: str):
        map_location = f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu'
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        self.model.module.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        if self.scheduler and ckpt['scheduler']:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        if self.scaler and ckpt['scaler']:
            self.scaler.load_state_dict(ckpt['scaler'])
    @abstractmethod
    def train_epoch(self, epoch):
        """
        train_progress_bar =tqdm(
            self.train_loader,
            desc=f"Train Epoch {epoch + 1}/{self.cfg.n_epochs}",
            disable=(self.rank != 0),
        )
        for i,batch in enumerate(train_progress_bar):
            if self.cfg.demo:
                break
        :param epoch:
        :return: train log dict
        """
        pass
    @abstractmethod
    def valid_epoch(self):
        """

        :return: train log dict
        """
        pass
    @abstractmethod
    def test_epoch(self):
        """

        :return: test log dict
        """
        pass
    @abstractmethod
    def custom_task(self, *args,**kwargs):
        """
        自定义任务
        :param args:
        :param kwargs:
        :return: task log dict
        """
        pass

    def __enter__(self):
        self.setup()
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        dist.destroy_process_group()