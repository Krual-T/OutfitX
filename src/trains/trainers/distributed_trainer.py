import datetime
import os
import pathlib
from abc import ABC, abstractmethod
import logging
import random
from typing import Dict, Any, Literal

import numpy as np
import torch
import torch.distributed as dist

from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

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

        self.local_rank = None
        self.rank = None
        self.world_size = None
        # self.num_workers = cfg.n_workers_per_gpu
        # self.batch_size = cfg.batch_sz_per_gpu

        self.model :nn.Module = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.logger = None
        self.wandb_run = None
        self._entered = False
        self.train_loader:DataLoader = None
        self.valid_loader:DataLoader = None
        self.test_loader:DataLoader = None

    def run(self):
        # FIXME 死锁
        """按 cfg.n_epochs 迭代，自动执行 train/valid/test/custom_task，并在每轮末同步与保存检查点。"""
        if not self._entered:
            raise RuntimeError("需在 with 语句中使用 DistributedTrainer。")

        for epoch in range(self.cfg.n_epochs):
            try:
                # 可选：在训练阶段设置随机种子
                if self.train_loader and hasattr(self.train_loader.sampler, 'set_epoch'):
                    self.train_loader.sampler.set_epoch(epoch)

                # 训练阶段
                if self.train_loader:
                    self.model.train()
                    self.train_epoch(epoch)
                    if self.scheduler:
                        self.scheduler.step()

                # 验证阶段
                if self.valid_loader:
                    self.model.eval()
                    self.valid_epoch()

                # 测试阶段
                if self.test_loader:
                    self.model.eval()
                    self.test_epoch()

                # 自定义任务
                self.custom_task(epoch=epoch)

            except Exception as e:
                # 捕获异常，上报并广播至所有进程
                msg = f"Epoch {epoch} 异常: {e}"
                print(msg)
                self.log(msg, level="error")
                _ = self.broadcast_termination_signal(encountered=True)
                raise e

            # 检查是否有其他进程请求终止
            if self.broadcast_termination_signal():
                self.log("收到终止信号，退出训练", level="warning")
                break

            # 全局同步：等待所有进程到此点，确保 checkpoint 存储时一致
            dist.barrier()

            # 保存检查点（仅在 rank0）
            if self.rank == 0 and self.train_loader and self.cfg.auto_save_checkpoint:
                self.save_checkpoint(epoch)
                # self.load_checkpoint(checkpoint_path)

    def setup_logger(self):
        # 只有 rank 0 创建日志文件
        if self.rank == 0:
            log_dir = self.cfg.LOG_DIR or pathlib.Path("./logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_name = f"{self.cfg.project_name + (('_' + self.cfg.name) if self.cfg.name else '')}.log"
            log_file = log_dir / log_name

            # 配置日志记录器
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',  # 定义日志输出格式
                handlers=[  # 通过 handlers 来定义输出方式
                    logging.FileHandler(log_file),  # 将日志输出到文件
                    logging.StreamHandler()  # 默认将日志输出到控制台
                ]
            )

            self.logger = logging.getLogger()  # 获取根日志记录器

            # 初始化WandB
            if self.cfg.wandb_key is not None:  # 只有rank=0的进程需要初始化WandB
                import wandb
                wandb.login(key=self.cfg.wandb_key)
                self.wandb_run = wandb.init(
                    project=self.cfg.project_name,
                    config=self.cfg.__dict__,
                    name=self.cfg.name
                )
    def setup_seed(self):
        random.seed(self.cfg.seed)
        os.environ['PYTHONHASHSEED'] = str(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setup_ddp_env(self):
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend=self.cfg.backend, init_method="env://")

    def setup(self):
        if not self._entered:
            raise RuntimeError("DistributedTrainer must be used in a 'with' block")
        setup_completed = lambda msg: self.log(f"{msg} 初始化完成",level="info")
        setup_failed = lambda msg: self.log(f"{msg} 初始化失败", level="error")
        not_setup = lambda msg: self.log(f"{msg} 未初始化",level="warning")
        # 初始化日志记录器
        try:
            self.setup_logger()
            setup_completed("logger")
        except Exception as e:
            print("logger 初始化失败")
            raise e
        # 初始化随机种子
        try:
            self.setup_seed()
            setup_completed("seed")
        except Exception as e:
            setup_failed("seed")
            raise e
        # 初始化分布式环境
        try:
            self.setup_ddp_env()
            setup_completed("ddp_environment")
        except Exception as e:
            setup_failed("ddp_environment")
            raise e
        # 初始化模型
        try:
            model = self.load_model()
            if model is None:
                raise ValueError("fn: load_model() must return a model")
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                model.cuda(self.local_rank)
                self.model: DDP = DDP(model, device_ids=[self.local_rank])
            else:
                self.model: DDP = DDP(model)
            setup_completed("model")
        except Exception as e:
            setup_failed("model")
            raise e
        # 初始化优化器
        try:
            self.optimizer = self.load_optimizer()
            if self.optimizer is None:
                raise ValueError("fn: load_optimizer() must return an optimizer")
            setup_completed("optimizer")
        except Exception as e:
            setup_failed("optimizer")
            raise e
        # 初始化学习率调节器
        try:
            self.scheduler = self.load_scheduler()
            if self.scheduler is not None:
                setup_completed("scheduler")
            else:
                not_setup("scheduler")
        except Exception as e:
            setup_failed("scheduler")
            raise e
        # 初始化scaler
        try:
            self.scaler = self.load_scaler()
            if self.scaler is not None:
                setup_completed("scaler")
            else:
                not_setup("scaler")
        except Exception as e:
            setup_failed("scaler")
            raise e
        # 初始化dataloader
        try:
            self.setup_dataloaders()
            not_setup_sampler = lambda loader: self.log(f"{loader}初始化完成,但{loader}.sampler 未初始化", level="warning")

            if self.train_loader is None:
                not_setup("train_loader")
            elif self.train_loader.sampler is None:
                not_setup_sampler("train_loader")
            else:
                setup_completed("train_loader")

            if self.valid_loader is None:
                not_setup("valid_loader")
            elif self.valid_loader.sampler is None:
                not_setup_sampler("valid_loader")
            else:
                setup_completed("valid_loader")

            if self.test_loader is None:
                not_setup("test_loader")
            elif self.test_loader.sampler is None:
                not_setup_sampler("test_loader")
            else:
                setup_completed("test_loader")

        except Exception as e:
            setup_failed("data_loaders")
            raise e

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

    def load_checkpoint(self, ckpt_path: str):
        map_location = f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu'
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        self.model.module.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        if self.scheduler and ckpt['scheduler']:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        if self.scaler and ckpt['scaler']:
            self.scaler.load_state_dict(ckpt['scaler'])

    def log(self,
        msg: str = None,
        logs: Dict[str:Any] = None,
        level: Literal['info', 'warning', 'error'] = 'info'
    ):
        if self.rank == 0:
            set_log = {
                'info': self.logger.info,
                'warning': self.logger.warning,
                'error': self.logger.error
            }
            set_log[level](msg)

            if self.wandb_run is not None and logs is not None:
                self.wandb_run.log(logs)
    def broadcast_termination_signal(self, encountered: bool=False) -> bool:
        """
        各进程调用时传入自身是否遇到异常(encountered=True/False)，
        使用 all_reduce 将所有进程的异常标志合并（取最大），
        若最终为 True，则所有进程应优雅退出。
        """
        flag = torch.tensor([1 if encountered else 0], device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

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
        self._entered = True
        env_required = ("LOCAL_RANK", "RANK", "WORLD_SIZE")
        error_msg = (
"""
请使用 torchrun （recommend）命令运行程序, 例如: 
    torchrun --nproc_per_node=4 --master_port=12345(one node is optional) main.py
或者使用 python -m torch.distributed.launch 命令运行程序, 例如: 
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345(one node is optional) main.py
"""
        )
        if not all(k in os.environ for k in env_required):
            raise RuntimeError(error_msg)

        self.setup()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # 若在主进程捕获到异常，可广播终止信号
            self.broadcast_termination_signal(True)
            # 只有出错的rank会输出
            time = datetime.datetime.now()
            print(f"{time} error:rank{self.rank}")
        # 清理分布式环境
        dist.destroy_process_group()

        if self.rank==0 and self.cfg.auto_save_checkpoint:
            self.save_checkpoint(-1)

