import pickle
from math import ceil

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from typing import Optional, Literal, cast, Union
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist
from src.losses import FocalLoss
from src.models import OutfitTransformer
from src.models.configs import OutfitTransformerConfig
from src.models.processor.outfit_transformer.outfit_transformer_original_compatibility_prediction_task_processor import \
    OutfitTransformerOriginalCompatibilityPredictionTaskProcessor
from src.trains.configs.compatibility_prediction_train_config import CompatibilityPredictionTrainConfig
from src.trains.datasets import PolyvoreItemDataset
from src.trains.datasets.polyvore.polyvore_compatibility_dataset import PolyvoreCompatibilityPredictionDataset
from src.trains.trainers.distributed_trainer import DistributedTrainer

class OriginalCompatibilityPredictionTrainer(DistributedTrainer):

    def __init__(self, cfg:Optional[CompatibilityPredictionTrainConfig]=None, run_mode:Literal['train-valid', 'test', 'custom']= 'train-valid'):
        if cfg is None:
            cfg = CompatibilityPredictionTrainConfig(
                batch_size=85,
                broadcast_buffers=False,
                accumulation_steps= 10,
                dataloader_workers=12
            )
        super().__init__(cfg=cfg, run_mode=run_mode)
        self.cfg = cast(CompatibilityPredictionTrainConfig, cfg)
        self.loss:Union[FocalLoss,None] = None
        self.device_type = None
        self.model_cfg = OutfitTransformerConfig()
        self.processor = OutfitTransformerOriginalCompatibilityPredictionTaskProcessor(self.model_cfg)
        self.best_metrics = {
            'AUC': 0.0,
            'Precision': 0.0,
            'Recall': 0.0,
            'F1': 0.0,
            'Accuracy': 0.0,
            'loss': np.inf,
        }
    def input_dict_to_device(self, batch_dict:dict):
        input_dict = batch_dict['input_dict']
        input_dict['outfit_mask'] = input_dict['outfit_mask'].to(self.local_rank,non_blocking=True)
        input_dict['encoder_input_dict']['images'] = input_dict['encoder_input_dict']['images'].to(self.local_rank,non_blocking=True)
        input_dict['encoder_input_dict']['texts'] = {
            k: v.to(self.local_rank,non_blocking=True) for k, v in input_dict['encoder_input_dict']['texts'].items()
        }
        return input_dict

    def train_epoch(self, epoch: int) -> None:
        # torch.backends.cuda.enable_flash_sdp(False)
        # torch.backends.cuda.enable_math_sdp(True)
        # torch.backends.cuda.enable_mem_efficient_sdp(False)
        self.model.train()
        if hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(epoch)
        train_processor = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.cfg.n_epochs}") if self.rank == 0 else self.train_dataloader
        self.optimizer.zero_grad()
        local_total_loss = torch.tensor(0.0, device=self.local_rank, dtype=torch.float32)
        local_y_hats = []
        local_labels = []
        for step,batch_dict in enumerate(train_processor):
            with autocast(enabled=self.cfg.use_amp, device_type=self.device_type):
                input_dict = self.input_dict_to_device(batch_dict)
                y_hats = self.model(**input_dict).squeeze(dim=-1)
                labels = batch_dict['label'].to(self.local_rank,non_blocking=True)
                loss = self.loss(y_hat=y_hats, y_true=labels)
                original_loss = loss.clone().detach()
                loss = loss / self.cfg.accumulation_steps

            self.scaler.scale(loss).backward()
            update_grad = ((step + 1) % self.cfg.accumulation_steps == 0) or ((step+1) == len(self.train_dataloader))
            if update_grad:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
            # if self.world_size > 1:
            #     dist.barrier()
            # metrics = self.build_metrics(
            #         local_y_hats=y_hats.detach(),
            #         local_labels=labels.detach(),
            #         local_loss=original_loss,
            #         epoch=epoch,
            # )
            # metrics = {
            #     'learning_rate': self.scheduler.get_last_lr()[0] if self.scheduler else self.cfg.learning_rate,
            #     **metrics
            # }
            # metrics = {f'train/batch/{k}': v for k, v in metrics.items()}
            # metrics = {
            #     'batch_step': epoch * len(self.train_dataloader) + step,
            #     **metrics
            # }
            # self.log(
            #     level='info',
            #     msg=str(metrics),
            #     metrics=metrics
            # )

            local_total_loss += original_loss
            local_y_hats.append(y_hats.detach())
            local_labels.append(labels.detach())
            # batch end--------------------------------------------------------------------------------------------------------

        local_y_hats = torch.cat(local_y_hats, dim=0)
        local_labels = torch.cat(local_labels, dim=0)
        # if self.world_size > 1:
        #     dist.barrier()
        #
        metrics = self.build_metrics(
            local_y_hats=local_y_hats,
            local_labels=local_labels,
            local_loss=local_total_loss,
            batch_count=len(self.train_dataloader),
            epoch=epoch+1
        )
        metrics = {f'{k}/train/epoch': v for k, v in metrics.items()}
        metrics = {
            'epoch':epoch+1,
            **metrics
        }
        self.log(
            level='info',
            msg=f"Epoch {epoch+1}/{self.cfg.n_epochs} -->Train End \n {str(metrics)}",
            metrics=metrics
        )

    @torch.no_grad()
    def valid_epoch(self, epoch: int):
        self.model.eval()
        valid_processor = tqdm(self.valid_dataloader, desc=f"Epoch {epoch+1}/{self.cfg.n_epochs}") if self.rank == 0 else self.valid_dataloader
        local_total_loss = torch.tensor(0.0, device=self.local_rank, dtype=torch.float32)
        local_y_hats = []
        local_labels = []
        for step,batch_dict in enumerate(valid_processor):
            with autocast(device_type=self.device_type, enabled=self.cfg.use_amp):
                input_dict = self.input_dict_to_device(batch_dict)
                y_hats = self.model(**input_dict).squeeze(dim=-1)
                labels = batch_dict['label'].to(self.local_rank)
                loss = self.loss(y_hat=y_hats, y_true=labels)
                original_loss = loss.clone().detach()
            # metrics = self.build_metrics(
            #         local_y_hats=y_hats.detach(),
            #         local_labels=labels.detach(),
            #         local_loss=original_loss,
            #         epoch=epoch,
            # )
            # metrics = {f'valid/batch/{k}': v for k, v in metrics.items()}
            # metrics = {
            #     'batch_step': epoch * len(self.valid_dataloader) + step,
            #     **metrics
            # }
            # self.log(
            #     level='info',
            #     msg=str(metrics),
            #     metrics=metrics
            # )

            local_total_loss += original_loss
            local_y_hats.append(y_hats.detach())
            local_labels.append(labels.detach())
            # batch end--------------------------------------------------------------------------------------------------------

        local_y_hats = torch.cat(local_y_hats, dim=0)
        local_labels = torch.cat(local_labels, dim=0)

        metrics = self.build_metrics(
            local_y_hats=local_y_hats,
            local_labels=local_labels,
            local_loss=local_total_loss,
            batch_count=len(self.valid_dataloader),
            epoch=epoch
        )

        self.maybe_save_best_models(metrics=metrics, epoch=epoch)

        metrics = {f'{k}/valid/epoch': v for k, v in metrics.items()}
        metrics = {
            'epoch':epoch+1,
            **metrics
        }
        self.log(
            level='info',
            msg=f"Epoch {epoch+1}/{self.cfg.n_epochs} --> Valid End \n\n {str(metrics)} \n",
            metrics=metrics
        )

    @torch.no_grad()
    def test(self):
        ckpt_name_prefix = self.model_cfg.model_name
        ckpt_path = self.cfg.checkpoint_dir / f'{ckpt_name_prefix}_best_AUC.pth'
        # ckpt_path = self.cfg.checkpoint_dir.parent / 'complementary_item_retrieval' /f'{ckpt_name_prefix}_best_Recall@1.pth'
        self.load_checkpoint(ckpt_path=ckpt_path, only_load_model=True)
        self.model.eval()
        test_processor = tqdm(self.test_dataloader, desc='[Test] Compatibility Prediction')
        all_y_hats = []
        all_labels = []
        for step, batch_dict in enumerate(test_processor):
            with autocast(enabled=self.cfg.use_amp, device_type=self.device_type):
                input_dict = self.input_dict_to_device(batch_dict)
                y_hats = self.model(**input_dict).squeeze(dim=-1)
            labels = batch_dict['label']
            all_y_hats.append(y_hats.detach())
            all_labels.append(labels.detach())
            # metrics = self.compute_cp_metrics(y_hats=all_y_hats[-1], labels=all_labels[-1])
            # metrics = {f'test/batch/{k}': v for k, v in metrics.items()}
            # metrics = {
            #     'batch_step': step,
            #     **metrics
            # }
            # self.log(
            #     level='info',
            #     msg=str(metrics),
            #     metrics=metrics
            # )
            # test_processor.set_postfix(**metrics)

        all_y_hats = torch.cat(all_y_hats, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        metrics = self.compute_cp_metrics(y_hats=all_y_hats, labels=all_labels)
        metrics = {f'{k}/test': v for k, v in metrics.items()}
        self.log(
            level='info',
            msg=f"[Test] Compatibility --> Results:\n\n {str(metrics)} \n",
            metrics=metrics
        )

    def setup_train_and_valid_dataloader(self):
        prefix = f"{self.model_cfg.model_name}_{PolyvoreItemDataset.embed_file_prefix}"
        train_dataset = PolyvoreCompatibilityPredictionDataset(
            polyvore_type=self.cfg.polyvore_type,
            mode='train',
            dataset_dir=self.cfg.dataset_dir,
            load_image=True,
            load_image_tensor=True
        )

        valid_dataset = PolyvoreCompatibilityPredictionDataset(
            polyvore_type=self.cfg.polyvore_type,
            mode='valid',
            dataset_dir=self.cfg.dataset_dir,
            load_image=True,
            load_image_tensor = True
        )

        # ✅ 根据是否分布式，动态选择 sampler
        if self.world_size > 1:
            train_sampler = DistributedSampler(
                dataset=train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True
            )
            valid_sampler = DistributedSampler(
                dataset=valid_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,  # 🚀 建议验证集也shuffle
                drop_last=False
            )
            shuffle_flag = False  # 有 sampler 时，不要再设置 shuffle
        else:
            train_sampler = None
            valid_sampler = None
            shuffle_flag = True
        self.train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle_flag if train_sampler is None else False,
            sampler=train_sampler,
            num_workers=self.cfg.dataloader_workers,
            pin_memory=True,
            persistent_workers=True if self.cfg.dataloader_workers > 0 else False,
            collate_fn=self.processor,
            prefetch_factor=2,
        )

        self.valid_dataloader = DataLoader(
            dataset=valid_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            sampler=valid_sampler,
            num_workers=self.cfg.dataloader_workers,
            pin_memory=True,
            persistent_workers=True if self.cfg.dataloader_workers > 0 else False,
            collate_fn=self.processor,
            prefetch_factor=2,
        )

    def setup_test_dataloader(self):
        prefix = f"{self.model_cfg.model_name}_{PolyvoreItemDataset.embed_file_prefix}"
        # item_embeddings = self.load_embeddings(embed_file_prefix=prefix)
        test_dataset = PolyvoreCompatibilityPredictionDataset(
            polyvore_type=self.cfg.polyvore_type,
            mode='test',
            dataset_dir=self.cfg.dataset_dir,
            # embedding_dict=item_embeddings,
            load_image=True,
            load_image_tensor=True
        )
        self.test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.dataloader_workers,
            collate_fn=self.processor
        )

    def hook_after_setup(self):
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.world_size >1 and self.run_mode == 'test':
            raise ValueError("测试模式下不支持分布式")

    def load_model(self) -> nn.Module:
        return OutfitTransformer(cfg=self.model_cfg)

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

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        return {item_id: embedding for item_id, embedding in zip(all_ids, all_embeddings)}

    def load_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate)

    def load_scheduler(self):
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            max_lr=self.cfg.learning_rate,
            epochs=self.cfg.n_epochs,
            steps_per_epoch=ceil(len(self.train_dataloader) / self.cfg.accumulation_steps),
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=1e4
    )

    def load_scaler(self):
        return torch.amp.GradScaler()

    def load_loss(self):
        return FocalLoss(alpha=0.75, gamma=2, reduction='mean')

    def build_metrics(
        self,
        local_y_hats:torch.Tensor,
        local_labels:torch.Tensor,
        local_loss:torch.Tensor,
        epoch:int,
        batch_count:int = 1,
    ):
        local_y_hats = local_y_hats.detach()
        local_labels = local_labels.detach()
        local_loss = local_loss.detach()
        # 一般不会很大，所以可以直接 gather 到当前进程
        all_y_hats = [torch.empty_like(local_y_hats) for _ in range(self.world_size)]
        all_labels = [torch.empty_like(local_labels) for _ in range(self.world_size)]
        all_loss = [torch.empty_like(local_loss) for _ in range(self.world_size)]
        if self.world_size > 1:
            dist.all_gather(all_y_hats, local_y_hats)
            dist.all_gather(all_labels, local_labels)
            dist.all_gather(all_loss, local_loss)
        else:
            all_y_hats[0] = local_y_hats
            all_labels[0] = local_labels
            all_loss[0] = local_loss

        all_y_hats = torch.cat(all_y_hats, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_loss = torch.stack(all_loss).mean()/batch_count

        metrics = self.compute_cp_metrics(y_hats=all_y_hats, labels=all_labels)
        return {
            'loss': all_loss.item(),
            **metrics
        }
    def compute_cp_metrics(self, y_hats: torch.Tensor, labels: torch.Tensor):
        # 确保在 CPU 上处理 numpy 运算
        probs = torch.sigmoid(y_hats.float()).detach().cpu()
        labels = labels.int().detach().cpu()
        predictions = probs.clone()

        # 🎯 AUC 单独计算（注意要做异常判断）
        auc = roc_auc_score(labels.numpy(), predictions.numpy()) if len(torch.unique(labels)) > 1 else 0.0

        # 🎯 离散化概率 -> 二值预测
        predictions = (predictions > 0.5).int()

        # 🎯 手动统计 TP / FP / FN
        tp = torch.sum((predictions == 1) & (labels == 1)).item()
        fp = torch.sum((predictions == 1) & (labels == 0)).item()
        fn = torch.sum((predictions == 0) & (labels == 1)).item()

        # 🎯 计算指标
        accuracy = torch.mean((predictions == labels).float()).item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # ✅ 返回最终结果（键名保持一致性）
        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUC': auc
        }
    # def compute_cp_metrics(self, y_hats: torch.Tensor, labels: torch.Tensor):
    #     # 重置 metric 状态，避免跨 epoch 累积污染
    #     self.auroc_metric.reset()
    #     self.precision_metric.reset()
    #     self.recall_metric.reset()
    #     self.f1_metric.reset()
    #     self.accuracy_metric.reset()
    #
    #     # 确保输入在同一设备
    #     y_hats = y_hats.to(self.local_rank)
    #     labels = labels.to(self.local_rank)
    #
    #     # ✅ 1. 准备好不同任务需要的格式
    #     # 概率输出（用于 AUC）
    #     probs = torch.sigmoid(y_hats.float())  # logits → probability
    #
    #     # 离散预测（用于 F1、Recall 等）
    #     preds = (probs > 0.5).int()
    #
    #     # 标签必须是 int（不能是 float），否则 torchmetrics 会行为异常
    #     labels_int = labels.int()
    #     if torch.distributed.get_rank() == 0:
    #         bins = [0.2, 0.3, 0.4, 0.5]
    #         count_0_0_2 = (probs <= 0.2).sum().item()
    #         count_0_2_0_3 = ((probs > 0.2) & (probs <= 0.3)).sum().item()
    #         count_0_3_0_4 = ((probs > 0.3) & (probs <= 0.4)).sum().item()
    #         count_0_4_0_5 = ((probs > 0.4) & (probs <= 0.5)).sum().item()
    #         count_above_0_5 = (probs > 0.5).sum().item()
    #
    #         print(f"[🔍 Prediction Probability Distribution]")
    #         print(f"  <=0.2     : {count_0_0_2}")
    #         print(f"  0.2~0.3   : {count_0_2_0_3}")
    #         print(f"  0.3~0.4   : {count_0_3_0_4}")
    #         print(f"  0.4~0.5   : {count_0_4_0_5}")
    #         print(f"  >0.5      : {count_above_0_5}")
    #
    #     # ✅ 2. 计算各类指标
    #     auc = self.auroc_metric(probs, labels_int)
    #     precision = self.precision_metric(preds, labels_int)
    #     recall = self.recall_metric(preds, labels_int)
    #     f1 = self.f1_metric(preds, labels_int)
    #     accuracy = self.accuracy_metric(preds, labels_int)
    #
    #     # ✅ 3. 转为 Python float，方便 wandb / JSON 等记录
    #     return {
    #         'Accuracy': accuracy.item(),
    #         'Precision': precision.item(),
    #         'Recall': recall.item(),
    #         'F1': f1.item(),
    #         'AUC': auc.item()
    #     }

    def maybe_save_best_models(self, metrics: dict, epoch: int):
        if self.rank != 0:
            return
        for metric,metric_value in metrics.items():
            if metric !='AUC' and metric!='loss':
                continue
            sign = 1 if metric=='loss' else -1
            best = self.best_metrics.get(metric, sign * np.inf)
            if metric_value * sign < best * sign:
                self.best_metrics[metric] = metric_value
                ckpt_name = f"{self.model_cfg.model_name}_best_{metric}"
                self.save_checkpoint(ckpt_name=ckpt_name,epoch=epoch, model_cfg_dict=self.model_cfg.__dict__)
                self.log(
                    level='info',
                    msg=f"✅ New best {metric}: {metric_value:.4f}, saved as {ckpt_name}.pth"
                )

    def setup_custom_dataloader(self):
        pass
    def custom_task(self, *args, **kwargs):
        pass