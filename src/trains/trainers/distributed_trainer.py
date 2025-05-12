import os
import pathlib
import logging
import random
from contextlib import contextmanager
from functools import wraps

import numpy as np
import torch

import torch.distributed as dist
from abc import ABC, abstractmethod
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from typing import Dict, Any, Literal, final
from src.trains.configs import BaseTrainConfig


class DistributedTrainer(ABC):
    """
    必须在 'with' 语句中使用，并运行命令：

        - torchrun --nproc_per_node=4 --master_port=12345 main.py

        - torchrun --standalone --nproc_per_node=xxx main.py
    ---------------------------------------------------------------------------------------------------------------------------------------------

    :example：
        with DistributedTrainer(cfg) as trainer:
            trainer.run()

    ---------------------------------------------------------------------------------------------------------------------------------------------

    Parameters:
        run_mode: str

            - 'train-valid': 仅执行 train_epoch 和 valid_epoch，不执行 test_epoch 和 custom_task。方法中包含关于 train/valid 的分布式训练提醒。

            - 'test': 仅执行测试（无轮次）

            - 'custom': 自定义任务（无轮次）

            通常以上模式可以满足需求，可以自定义实现 'custom_task' 方法（无轮次）或重写 run 方法，但仍建议参考我的 run 方法实现。

        cfg: BaseTrainConfig(包含训练过程所需的配置项)：

            - project_name: 项目名称

            - name: 实验名称（可选）

            - seed: 随机种子

            - n_epochs: 总训练轮数

            - batch_size: 每批次样本数

            - auto_save_checkpoint: 是否自动保存检查点

            - checkpoint_dir: 保存检查点的目录

            - LOG_DIR: 日志目录（可选）

            - wandb_key: 用于登录 WandB 的 API 密钥（可选）

            - backend: 分布式训练所使用的后端（如 NCCL、Gloo 等）

            - find_unused_parameters: 是否查找未使用的参数

            - 其他自定义配置项（例如优化器参数、调度器配置等）

    注意：
        - 抽象方法并非都需要实现，具体实现根据需求决定。
        - setup 方法会根据 cfg 中的配置，初始化训练过程所需的各种组件，如日志记录器、分布式环境、模型、优化器、学习率调节器等。

    关于 setup 和 load 开头的方法：
        - setup 开头的方法不会返回值，而是直接注册到 self 中。
        - load 开头的方法会返回一个对象，并自动注册到 self 中。

    关于日志：
        - setup_logger 支持 wandb（当 cfg.WANDB_KEY 不为 None）以及本地日志记录，具体请查看方法文档。
        - 提供 self.log 方法用于打印日志（支持 wandb），具体请查看方法文档。
        - 如果您有特殊的日志需求，建议自行重写 setup_logger 和 log 方法（但不推荐）。

    强制要求定义的几个 DataLoader 名称，请根据run_mode初始化对应的dataloader，
    提供setup_train_and_valid_dataloader方法，和setup_test_dataloader方法，
    以及setup_custom_dataloader方法（钩子）。
        - self.train_dataloader: DataLoader = None
        - self.valid_dataloader: DataLoader = None
        - self.test_dataloader: DataLoader = None
        - 其他需求在钩子函数（setup_custom_dataloader）中实现。

    关于分布式的一些注意点：
        死锁:分布式训练中由于一些方法会阻塞进程，当出现一些进程达到同步点而其他经常因为某些原因，
        例如条件限制（不要在if中阻塞进程！！！）、进程出错、进程被中断等情况，可能会导致进程无法继续执行，
        此时达到同步点的进程会被持续阻塞，导致死锁。

        解决方法：
            1. 避免在 if 等条件语句中阻塞进程，违反了分布式训练的原则（确保所有进程都能达到同步点）。
            2. 捕获异常并继续raise异常并广播到所有进程，并终止全部进程（就像safe_process_context，一个上下文管理器的实现那样做）

        幸运的是，本架构对于每一个epoch执行的任务（running_epoch方法）提供了上下文管理机制（safe_process_context），
        这意味着你不用太担心繁琐的try except 语句，只需要你遵循分布式原则并在你的代码中抛出异常即可（不要except异常却不raise异常）。
        如果我的实现无法满足你的日志记录或其他需求，你可以按照我的实现思路自行实现。

        以下是一些常见的会造成阻塞的分布式方法：

            - dist.barrier()：同步所有进程
            - dist.reduce()、dist.all_reduce()：数据聚合操作
            - dist.broadcast()：广播数据（不推荐）
            - dist.all_gather_object()：收集所有数据后推送到所有进程(推荐) 通常出现在train_epoch中，用于聚合每个进程的损失值（日志记录）
            - dist.scatter()、dist.gather()：数据分发和收集
            - dist.send()、dist.recv()：点对点通信
            - dist.all_to_all()：多对多通信
            - dist.wait()：等待操作完成

    api:
        - self.build_error_msg: 用于构建错误信息，具体请查看方法文档。
        - self.save_checkpoint: 用于保存检查点，具体请查看方法文档。
        - self.load_checkpoint: 用于加载检查点，具体请查看方法文档。
        - self.log: 用于打印日志，具体请查看方法文档。
        - self.set_log_: 用于设置日志级别（log内部调用），具体请查看方法文档。
        - safe_process_context: 用于包裹可能出错的代码块（不要包裹带阻塞的dist方法），用于捕获异常并广播到所有进程，具体请查看方法文档。

    内部所有属性：
        - self.cfg = cfg
        - self.run_mode = run_mode
        - self.local_rank = None
        - self.rank = None
        - self.world_size = None
        - self.model: nn.Module = None
        - self.optimizer = None
        - self.scheduler = None
        - self.scaler = None
        - self.logger = None
        - self.wandb_run = None
        - self._entered = False
        - self.train_dataloader: DataLoader = None
        - self.valid_dataloader: DataLoader = None
        - self.test_dataloader: DataLoader = None

    内部所有方法：
        - setup_logger: 设置日志记录
        - setup_ddp_env: 设置分布式训练环境
        - setup_seed: 设置随机种子
        - setup: 初始化各个组件
        - save_checkpoint: 保存模型检查点
        - load_checkpoint: 加载模型检查点
        - build_error_msg: 构建错误信息
        - running_epoch: 根据 run_mode 执行相应的训练/验证/测试任务
        - run: 按照 cfg.n_epochs 进行训练迭代
        - log: 记录日志

    不推荐重写的方法：
        - __enter__
        - __exit__
        - __init__
        - setup
        - run
        - build_error_msg
        - log
    """
    def __init__(
        self,
        cfg:BaseTrainConfig,
        run_mode:Literal['train-valid','test','custom'] = 'custom'
    ):
        self.cfg = cfg
        self.run_mode = run_mode
        self.local_rank = None
        self.rank = None
        self.world_size = None

        self.model :nn.Module = None
        self.optimizer:torch.optim.Optimizer = None
        self.scheduler:torch.optim.lr_scheduler.LRScheduler = None
        self.scaler:torch.amp.GradScaler = None
        self.logger = None
        self.wandb_run = None
        self._entered = False
        self.train_dataloader:DataLoader = None
        self.valid_dataloader:DataLoader = None
        self.test_dataloader:DataLoader = None
        self.loss = None
        # self.dataloader_workers = cfg.dataloader_workers
        # self.batch_size = cfg.batch_size

    @contextmanager
    def safe_process_context(self, *args, **kwargs,):
        process_errs = [None] * self.world_size
        local_err_msg = None
        try:
            yield
        except (KeyboardInterrupt, Exception) as e:
            # 捕获异常，上报并广播至所有进程
            local_err_msg = self.build_error_msg(*args, e=e, **kwargs)
        finally:
            dist.all_gather_object(process_errs, local_err_msg)  # 阻塞并广播本地情况到所有进程
            for err in process_errs:
                if err is not None:
                    self.log(str(err))
                    raise Exception(err)
            dist.barrier()

    def run(self):
        """
        运行训练过程，根据 cfg.n_epochs 进行训练迭代。
        每个 epoch 都会执行 running_epoch 方法，根据 run_mode 执行相应的训练/验证/测试任务/自定义任务。
        :return:
        """
        if not self._entered:
            raise RuntimeError("需在 with 语句中使用 DistributedTrainer。")
        if self.run_mode == 'train-valid':
            for epoch in range(self.cfg.n_epochs):
                self.train_epoch(epoch)
                self.valid_epoch(epoch)
        elif self.run_mode == 'test':
            self.test()
        elif self.run_mode == 'custom':
            # 自定义任务
            self.custom_task()

    def build_error_msg(self, epoch: int, e: BaseException) -> str:
        """
        构建错误信息，包含异常类型、异常信息和堆栈跟踪信息。
        :param epoch: 当前 epoch 数
        :param e: 捕获的异常
        :return: error_msg:str
        """
        import traceback as tb_module
        tb_str = "".join(tb_module.format_exception(type(e), e, e.__traceback__))
        error_msg = f"[Rank {self.rank} | Epoch {epoch}] \n" + tb_str
        return error_msg

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
        seed = self.cfg.seed+self.rank
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setup_ddp_env(self):
        # 防御性编程：确保在使用前已经正确初始化
        self.cfg.backend = self.cfg.backend if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=self.cfg.backend, init_method="env://")

    def setup(self):
        if not self._entered:
            raise RuntimeError("DistributedTrainer must be run as torchrun command and used in a 'with' block  ")
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
        # 初始化分布式环境
        try:
            self.setup_ddp_env()
            setup_completed("ddp_environment")
        except Exception as e:
            setup_failed("ddp_environment")
            raise e
        # 初始化随机种子
        try:
            self.setup_seed()
            setup_completed("seed")
        except Exception as e:
            setup_failed("seed")
            raise e
        # 初始化模型
        try:
            model = self.load_model()
            if model is None:
                raise ValueError("fn: load_model() must return a model")
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                model.cuda(self.local_rank)
                self.model: DDP = DDP(
                    module=model,
                    find_unused_parameters=self.cfg.find_unused_parameters,
                    device_ids=[self.local_rank]
                )
            else:
                self.model: DDP = DDP(
                    module= model,
                    find_unused_parameters=self.cfg.find_unused_parameters
                )
            setup_completed("model")
        except Exception as e:
            setup_failed("model")
            raise e
        # 初始化loss
        try:
            self.loss=self.load_loss()
            if self.loss is not None:
                setup_completed("loss")
            elif self.run_mode == 'train-valid':
                raise ValueError("In the train-valid mode,the fn: load_loss() must return a loss(nn.moudle)")
            elif self.run_mode == 'custom':
                not_setup("loss")
        except Exception as e:
            setup_failed("loss")
            raise e
        # 初始化dataloader
        try:
            self.setup_dataloaders()
            not_setup_sampler = lambda loader: self.log(f"{loader}初始化完成,但{loader}.sampler 未初始化", level="warning")
            # train-valid mode
            if self.run_mode == 'train-valid':
                # train_dataloader
                if self.train_dataloader is None:
                    raise ValueError("In the train-valid mode,the fn: setup_dataloaders() must Register and initialize self.train_dataloader")
                elif self.train_dataloader.sampler is None:
                    not_setup_sampler("train_dataloader")
                else:
                    setup_completed("train_dataloader")
                # valid_dataloader
                if self.valid_dataloader is None:
                    raise ValueError("In the train-valid mode,the fn: setup_dataloaders() must Register and initialize self.valid_dataloader")
                elif self.valid_dataloader.sampler is None:
                    not_setup_sampler("valid_dataloader")
                else:
                    setup_completed("valid_dataloader")
            # test mode
            elif self.run_mode == 'test':
                # test_dataloader
                if self.test_dataloader is None:
                    raise ValueError("In the test mode,the fn: setup_dataloaders() must Register and initialize self.test_dataloader")
                elif self.test_dataloader.sampler is None:
                    not_setup_sampler("test_dataloader")
                setup_completed("test_dataloader")

        except Exception as e:
            setup_failed("data_loaders")
            raise e
        # 初始化优化器
        try:
            if self.run_mode == 'train-valid':
                self.optimizer = self.load_optimizer()
                if self.optimizer is None:
                    raise ValueError("In the train-valid mode,the fn: load_optimizer() must return an optimizer")
                setup_completed("optimizer")
            elif self.run_mode!= 'test':
                not_setup("optimizer")
        except Exception as e:
            setup_failed("optimizer")
            raise e
        # 初始化学习率调节器
        try:
            if self.run_mode == 'train-valid':
                self.scheduler = self.load_scheduler()
                if self.scheduler is None:
                    raise ValueError("In the train-valid mode,the fn: load_scheduler() must return a scheduler")
                setup_completed("scheduler")
            elif self.run_mode != 'test':
                not_setup("scheduler")
        except Exception as e:
            setup_failed("scheduler")
            raise e
        # 初始化scaler
        try:
            if self.run_mode == 'train-valid':
                self.scaler = self.load_scaler()
                if self.scaler is not None:
                    setup_completed("scaler")
                elif self.run_mode != 'test':
                    not_setup("scaler")
        except Exception as e:
            setup_failed("scaler")
            raise e
        self.hook_after_setup()

    @abstractmethod
    def hook_after_setup(self):
        pass

    def setup_dataloaders(self):
        if self.run_mode == 'train-valid':
            self.setup_train_and_valid_dataloader()
        elif self.run_mode == 'test':
            self.setup_test_dataloader()
        elif self.run_mode == 'custom':
            self.setup_custom_dataloader()

    def save_checkpoint(self, epoch, ckpt_name = None):
        """
        保存模型检查点，包括模型参数、优化器状态、学习率调节器状态、scaler状态。
        检查点文件名为 epoch_{epoch}.pth，保存在 cfg.checkpoint_dir 目录下。
        注意：
            1. 检查点文件会被保存到所有进程中，因此需要在所有进程中都能访问到(建议rank==0时调用)。
            2. 检查点文件会被保存到 cfg.checkpoint_dir 目录下，因此需要确保该目录存在。
            3. 如果使用rank==0调用，最好在if外调用dist.barrier()，以确保进程统一。
            4. 保存的检查点不是ddp模型，而是原始模型（没有module），可以直接被load_checkpoint加载。
        :param epoch:int
            当前训练轮数
        :return: checkpoint_path:str
            检查点文件路径
        """
        checkpoint_dir = self.cfg.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = checkpoint_dir / (f"epoch_{epoch}.pth" if ckpt_name is None else f"{ckpt_name}.pth")
        torch.save({
            'epoch': epoch,
            'config': self.model.module.cfg.__dict__,
            'model': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'scaler': self.scaler.state_dict() if self.scaler else None,
        }, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, ckpt_path: str,only_load_model:bool = False, *args, **kwargs):
        """
        加载模型检查点，包括模型参数、优化器状态、学习率调节器状态、scaler状态。
        注意：
            1. 检查点文件会被加载到所有进程中，因此需要在所有进程中都能访问到。
            2. 不要在with以外使用
        :param only_load_model: bool
        :param ckpt_path:str
        :param args:
        :param kwargs:
        :return:
        """
        map_location = f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu'
        ckpt = torch.load(ckpt_path,*args,map_location=map_location,**kwargs)
        self.model.module.load_state_dict(ckpt['model'])
        if not only_load_model:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(ckpt['scheduler'])
            if self.scaler is not None:
                self.scaler.load_state_dict(ckpt['scaler'])

    @final
    @property
    def set_log_(self):
        """
        就是一个类变量（Dict）
        用于设置日志级别，支持 wandb（当 cfg.WANDB_KEY 不为 None）以及本地日志记录。
        使用方式：
            self.set_log_['info']
        :return: {
                'info': self.logger.info,
                'warning': self.logger.warning,
                'error': self.logger.error
            }[key]
        """
        return {
                'info': self.logger.info,
                'warning': self.logger.warning,
                'error': self.logger.error
            }
    def log(self,
        msg: str = None,
        metrics: Dict[str, Any] = None,
        level: Literal['info', 'warning', 'error'] = 'info'
    ):
        """
        打印日志（rank==0），支持 wandb（当 cfg.WANDB_KEY 不为 None）以及本地日志记录。
        :param msg: str
            日志信息
        :param metrics: Dict[str, Any]
            日志信息中的指标（只会用于wandb，如果要写在日志中，请在本实现的基础上重写加入功能，因为没有提供钩子）
        :param level: Literal['info', 'warning', 'error']
            日志级别
        :return:
        """
        if self.rank == 0:

            self.set_log_[level](msg)

            if metrics is not None:

                if self.wandb_run is not None:
                    self.wandb_run.log(metrics)

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
    def setup_train_and_valid_dataloader(self):
        pass

    @abstractmethod
    def setup_test_dataloader(self):
        pass

    @abstractmethod
    def setup_custom_dataloader(self):
        pass

    @abstractmethod
    def load_loss(self):
        pass

    @abstractmethod
    def train_epoch(self, epoch):
        """
        ```python
        model.train()
        if hasattr(train_dataloader.sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)  # Ensure shuffling is done correctly

        running_loss = 0.0
        optimizer.zero_grad()  # Zero gradients before each epoch

        for batch_idx, (inputs, labels) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast():  # Automatic Mixed Precision (AMP)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Scale the loss and backward
            scaler.scale(loss).backward()

            # Perform gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                # Step the optimizer only after accumulation_steps
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()  # Update the learning rate （α）

            running_loss += loss.item()

        # After each epoch, print average loss and do validation
        print(f"Rank {rank}, Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_dataloader):.4f}")

        ```
        :param epoch:int
        :return:
        """
        pass

    @abstractmethod
    def valid_epoch(self, epoch):
        """
        记得使用@torch.no_grad()
        记得在valid_epoch中调用self.log()方法，将验证结果记录到日志中
        使用带阻塞的dist方法前，请将有可能出错的部分采用with safe_process_context(epoch)包裹
        :return:
        """
        pass

    @abstractmethod
    def test(self):
        """
        运行一些测试，例如测试模型性能
        :return:
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
"""
        )
        if not all(k in os.environ for k in env_required):
            raise RuntimeError(error_msg)
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.cfg.dataloader_workers = min(
            self.cfg.dataloader_workers,
            max(1, (os.cpu_count() // max(1,torch.cuda.device_count()))-1)
        )
        self.setup()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        dist.barrier()  # 确保所有进程都到达此处
        if not exc_type and self.rank==0 and self.train_dataloader is not None and self.cfg.auto_save_checkpoint:
            self.log(f"[Checkpoint] Saved to {self.save_checkpoint(-1)}")

        if self.wandb_run is not None and self.rank == 0:
            self.wandb_run.finish()
        # 清理分布式环境
        dist.destroy_process_group()