import os
import pickle
import time
from typing import Optional, Literal, cast

import numpy as np
import torch
from torch import nn
from torch.cpu.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist
from src.losses import FocalLoss
from src.models import OutfitTransformer
from src.models.configs import OutfitTransformerConfig
from src.models.utils.model_utils import flatten_seq_to_one_dim
from src.trains.configs.compatibility_train_config import CompatibilityTrainConfig
from src.trains.datasets.polyvore.polyvore_compatibility_dataset import PolyvoreCompatibilityDataset
from src.trains.trainers.distributed_trainer import DistributedTrainer
from torchmetrics.classification import (
    BinaryAUROC, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAccuracy
)
class CompatibilityTrainer(DistributedTrainer):

    def __init__(self,cfg:Optional[CompatibilityTrainConfig]=None, run_mode:Literal['train-valid', 'test', 'custom']='train-valid'):
        if cfg is None:
            cfg = CompatibilityTrainConfig()
        super().__init__(cfg=cfg, run_mode=run_mode)
        self.cfg = cast(CompatibilityTrainConfig,cfg)
        self.loss:FocalLoss = None
        # 所有指标提前初始化一次，可以放在模型或 Trainer 中复用
        self.auroc_metric:BinaryAUROC = None
        self.precision_metric:BinaryPrecision = None
        self.recall_metric:BinaryRecall = None
        self.f1_metric:BinaryF1Score = None
        self.accuracy_metric:BinaryAccuracy = None

    def train_epoch(self, epoch: int) -> None:
        # 记录epoch开始时间
        epoch_start_time = time.time()

        self.model.train()
        if hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(epoch)
        train_processor = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.cfg.n_epochs}")
        self.optimizer.zero_grad()
        local_total_loss = torch.tensor(0.0, device=self.local_rank, dtype=torch.float32)
        local_y_hats = []
        local_labels = []
        for step,(queries, labels) in enumerate(train_processor):
            # 记录每个batch的开始时间
            batch_start_time = time.time()
            with self.safe_process_context(epoch=epoch):
                labels = torch.tensor(
                    data=labels,
                    dtype=torch.float32,
                    device=self.local_rank
                )

                with autocast(enabled=self.cfg.use_amp):
                    y_hats = self.model(queries)
                    loss = self.loss(y_hat=y_hats, y_true=labels)
                    original_loss = loss.detach()
                    loss = loss / self.cfg.accumulation_steps

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                if (step + 1) % self.cfg.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                batch_end_time = time.time()
                local_batch_time = torch.tensor(batch_end_time - batch_start_time, device=self.local_rank, dtype=torch.float32)
            dist.barrier()
            metrics = self.build_metrics(
                    local_y_hats=y_hats.detach(),
                    local_labels=labels.detach(),
                    local_loss=original_loss,
                    local_batch_time=local_batch_time,
                    epoch=epoch,
            )
            metrics = {
                'step': epoch * len(self.train_dataloader) + step,
                'learning_rate': self.scheduler.get_last_lr()[0] if self.scheduler else self.cfg.learning_rate,
                **metrics
            }
            metrics = {f'train-batch:{k}': v for k, v in metrics.items()}
            self.log(
                level='info',
                msg=str(metrics),
                metrics=metrics
            )
            train_processor.set_postfix(**metrics)

            local_total_loss += original_loss
            local_y_hats.append(y_hats.detach())
            local_labels.append(labels.detach())
            # batch end--------------------------------------------------------------------------------------------------------

        local_y_hats = torch.cat(local_y_hats, dim=0)
        local_labels = torch.cat(local_labels, dim=0)
        # 记录epoch结束时间
        epoch_end_time = time.time()
        dist.barrier()

        # 计算epoch耗时
        local_epoch_time = torch.tensor(epoch_end_time - epoch_start_time, device=self.local_rank, dtype=torch.float32)
        metrics = self.build_metrics(
            local_y_hats=local_y_hats,
            local_labels=local_labels,
            local_loss=local_total_loss,
            local_epoch_time=local_epoch_time,
            batch_count=len(self.train_dataloader)
        )
        metrics = {f'train-epoch:{k}': v for k, v in metrics.items()}
        self.log(
            level='info',
            msg=f"Epoch {epoch+1}/{self.cfg.n_epochs} --> End \n {str(metrics)}",
            metrics=metrics
        )
        # TODO: 保存模型

    @torch.no_grad()
    def valid_epoch(self, epoch: int):
        # 记录epoch开始时间
        epoch_start_time = time.time()
        self.model.eval()
        valid_processor = tqdm(self.valid_dataloader, desc=f"Epoch {epoch+1}/{self.cfg.n_epochs}")
        local_total_loss = torch.tensor(0.0, device=self.local_rank, dtype=torch.float32)
        local_y_hats = []
        local_labels = []
        for step,(queries, labels) in enumerate(valid_processor):
            # 记录每个batch的开始时间
            batch_start_time = time.time()
            with self.safe_process_context(epoch=epoch):
                labels = torch.tensor(
                    data=labels,
                    dtype=torch.float32,
                    device=self.local_rank
                )
                with autocast(enabled=self.cfg.use_amp):
                    y_hats = self.model(queries)
                    loss = self.loss(y_hat=y_hats, y_true=labels)
                    original_loss = loss.detach()

                batch_end_time = time.time()
                local_batch_time = torch.tensor(batch_end_time - batch_start_time, device=self.local_rank, dtype=torch.float32)
            dist.barrier()
            metrics = self.build_metrics(
                    local_y_hats=y_hats.detach(),
                    local_labels=labels.detach(),
                    local_loss=original_loss,
                    local_batch_time=local_batch_time,
                    epoch=epoch,
            )
            metrics = {
                'step': epoch * len(self.train_dataloader) + step,
                **metrics
            }
            metrics = {f'valid-batch:{k}': v for k, v in metrics.items()}
            self.log(
                level='info',
                msg=str(metrics),
                metrics=metrics
            )
            valid_processor.set_postfix(**metrics)

            local_total_loss += original_loss
            local_y_hats.append(y_hats.detach())
            local_labels.append(labels.detach())
            # batch end--------------------------------------------------------------------------------------------------------

        local_y_hats = torch.cat(local_y_hats, dim=0)
        local_labels = torch.cat(local_labels, dim=0)
        # 记录epoch结束时间
        epoch_end_time = time.time()
        dist.barrier()

        # 计算epoch耗时
        local_epoch_time = torch.tensor(epoch_end_time - epoch_start_time, device=self.local_rank, dtype=torch.float32)
        metrics = self.build_metrics(
            local_y_hats=local_y_hats,
            local_labels=local_labels,
            local_loss=local_total_loss,
            local_epoch_time=local_epoch_time,
            batch_count=len(self.train_dataloader)
        )
        metrics = {f'valid-epoch:{k}': v for k, v in metrics.items()}
        self.log(
            level='info',
            msg=f"Epoch {epoch+1}/{self.cfg.n_epochs} --> End \n {str(metrics)}",
            metrics=metrics
        )

    @torch.no_grad()
    def test(self):
        pass

    def setup_train_and_valid_dataloader(self):
        item_embeddings = self.load_embeddings(embed_file_prefix="embedding_subset_")
        collate_fn = lambda batch:(
            [item[0] for item in batch],
            [item[1] for item in batch]
        )
        train_dataset = PolyvoreCompatibilityDataset(
            polyvore_type=self.cfg.polyvore_type,
            mode='train',
            dataset_dir=self.cfg.dataset_dir,
            embedding_dict=item_embeddings,
            load_image=self.cfg.load_image
        )
        train_sampler = DistributedSampler(
            dataset=train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=True
        )
        self.train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.cfg.batch_size,
            sampler=train_sampler,
            num_workers=self.cfg.dataloader_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        valid_dataset = PolyvoreCompatibilityDataset(
            polyvore_type=self.cfg.polyvore_type,
            mode='valid',
            dataset_dir=self.cfg.dataset_dir,
            embedding_dict=item_embeddings,
            load_image=self.cfg.load_image
        )
        valid_sampler = DistributedSampler(
            dataset=valid_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            drop_last=False
        )
        self.valid_dataloader = DataLoader(
            dataset=valid_dataset,
            batch_size=self.cfg.batch_size,
            sampler=valid_sampler,
            num_workers=self.cfg.dataloader_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

    def setup_test_dataloader(self):
        item_embeddings = self.load_embeddings(embed_file_prefix="embedding_subset_")
        collate_fn = lambda batch:(
            [item[0] for item in batch],
            [item[1] for item in batch]
        )
        test_dataset = PolyvoreCompatibilityDataset(
            polyvore_type=self.cfg.polyvore_type,
            mode='test',
            dataset_dir=self.cfg.dataset_dir,
            embedding_dict=item_embeddings,
            load_image=self.cfg.load_image
        )
        self.test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.dataloader_workers,
            collate_fn=collate_fn
        )

    def hook_after_setup(self):
        # 所有指标提前初始化一次，可以放在模型或 Trainer 中复用
        self.auroc_metric = BinaryAUROC().to(self.local_rank)  # 或 device
        self.precision_metric = BinaryPrecision().to(self.local_rank)
        self.recall_metric = BinaryRecall().to(self.local_rank)
        self.f1_metric = BinaryF1Score().to(self.local_rank)
        self.accuracy_metric = BinaryAccuracy().to(self.local_rank)

    def load_model(self) -> nn.Module:
        cfg = OutfitTransformerConfig()
        return OutfitTransformer(cfg=cfg)

    def load_embeddings(self,embed_file_prefix:str="embedding_subset_") -> dict:
        """
        合并所有 embedding_subset_{rank}.pkl 文件，返回包含完整 id 列表和嵌入矩阵的 dict。
        """
        embedding_dir = self.cfg.precomputed_embedding_dir
        prefix = embed_file_prefix
        files = sorted(embedding_dir.glob(f"{prefix}*.pkl"))
        if not files:
            raise FileNotFoundError(f"找不到任何文件: {prefix}*.pkl")

        all_ids = []
        all_embeddings = []

        for file in files:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                all_ids.extend(data['ids'])
                all_embeddings.append(data['embeddings'])

        merged_embeddings = np.concatenate(all_embeddings, axis=0)
        return {'ids': all_ids, 'embeddings': merged_embeddings}

    def load_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate)

    def load_scheduler(self):
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            max_lr=self.cfg.learning_rate,
            epochs=self.cfg.n_epochs,
            steps_per_epoch=len(self.train_dataloader) // self.cfg.accumulation_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=1e4
    )

    def load_scaler(self):
        return torch.amp.GradScaler()

    def load_loss(self):
        return FocalLoss(alpha=0.5, gamma=2, reduction='mean')

    def build_metrics(
        self,
        local_y_hats:torch.Tensor,
        local_labels:torch.Tensor,
        local_loss:torch.Tensor,
        local_time:torch.Tensor,
        epoch:int,
        batch_count:int = 1,
    ):
        with self.safe_process_context(epoch=epoch):
            local_y_hats = local_y_hats.detach()
            local_labels = local_labels.detach()
            local_loss = local_loss.detach()
            # 一般不会很大，所以可以直接 gather 到当前进程
            all_y_hats = [torch.empty_like(local_y_hats) for _ in range(self.world_size)]
            all_labels = [torch.empty_like(local_labels) for _ in range(self.world_size)]
            all_loss = [torch.empty_like(local_loss) for _ in range(self.world_size)]
            all_time = [torch.empty_like(local_time) for _ in range(self.world_size)]
        # maybe use all_gather_into_tensor ?
        dist.all_gather(all_y_hats, local_y_hats)
        dist.all_gather(all_labels, local_labels)
        dist.all_gather(all_loss, local_loss)
        dist.all_gather(all_time, local_time)

        all_y_hats = torch.cat(all_y_hats, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_loss = torch.cat(all_loss, dim=0).mean()/batch_count
        all_time = torch.cat(all_time, dim=0).mean()

        metrics = self.compute_cp_metrics(y_hats=all_y_hats, labels=all_labels)
        return {
            'loss': all_loss.item(),
            'time': all_time.item(),
            **metrics
        }

    def compute_cp_metrics(self, y_hats: torch.Tensor, labels: torch.Tensor):

        # 确保输入在同一个设备上
        y_hats = y_hats.to(self.local_rank)
        labels = labels.to(self.local_rank)

        # AUC 需要概率
        auc = self.auroc_metric(y_hats, labels)

        # 其他指标需要离散分类
        pred_labels = (y_hats > 0.5).int()

        # Precision, Recall, F1, Accuracy
        precision = self.precision_metric(pred_labels, labels)
        recall = self.recall_metric(pred_labels, labels)
        f1 = self.f1_metric(pred_labels, labels)
        accuracy = self.accuracy_metric(pred_labels, labels)

        # 转为 Python float 方便日志或 JSON 输出
        return {
            'Accuracy': accuracy.item(),
            'Precision': precision.item(),
            'Recall': recall.item(),
            'F1': f1.item(),
            'AUC': auc.item()
        }

    def setup_custom_dataloader(self):
        pass
    def custom_task(self, *args, **kwargs):
        pass