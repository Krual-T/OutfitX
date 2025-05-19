import pickle
from collections import defaultdict
from math import ceil
from typing import cast, Literal, List, Dict

import numpy as np
import torch
from torch import nn, autocast, dtype
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses import SetWiseRankingLoss
from src.models import OutfitTransformer
from src.models.configs import OutfitTransformerConfig
from src.models.datatypes import OutfitComplementaryItemRetrievalTask
from src.models.processor import OutfitTransformerProcessorFactory
from src.trains.datasets import PolyvoreItemDataset
from src.trains.datasets.polyvore.polyvore_complementary_item_retrieval_dataset import \
    PolyvoreComplementaryItemRetrievalDataset
from src.trains.trainers.distributed_trainer import DistributedTrainer
from src.trains.configs import ComplementaryItemRetrievalTrainConfig as CIRTrainConfig

class ComplementaryItemRetrievalTrainer(DistributedTrainer):
    def __init__(self, cfg:CIRTrainConfig= None,run_mode: Literal['train-valid', 'test', 'custom'] = 'train-valid'):
        if cfg is None:
            cfg = CIRTrainConfig()
        super().__init__(cfg=cfg, run_mode=run_mode)
        self.device_type = None
        self.best_metrics = {}
        self.model_cfg = OutfitTransformerConfig()
        self.sample_mode = None
        if self.run_mode == 'train-valid':
            self.train_processor = OutfitTransformerProcessorFactory.get_processor(
                run_mode='train',
                task=OutfitComplementaryItemRetrievalTask,
            )
            self.valid_processor = OutfitTransformerProcessorFactory.get_processor(
                run_mode='valid',
                task=OutfitComplementaryItemRetrievalTask,
            )
        elif self.run_mode == 'test':
            self.test_processor = OutfitTransformerProcessorFactory.get_processor(
                run_mode='test',
                task=OutfitComplementaryItemRetrievalTask,
            )

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
                if epoch == self.cfg.switch_to_hard_n_epochs:
                    self.setup_train_and_valid_dataloader(sample_mode='hard')
                self.train_epoch(epoch)
                self.valid_epoch(epoch)
        elif self.run_mode == 'test':
            self.test()
        elif self.run_mode == 'custom':
            # 自定义任务
            self.custom_task()

    def train_epoch(self, epoch):
        self.model.train()
        train_processor = tqdm(self.train_dataloader, desc=f"Epoch {epoch}/{self.cfg.n_epochs}")
        self.optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=self.local_rank, dtype=torch.float)
        for step,batch_dict in enumerate(train_processor):
            input_dict = {
                k: (v if k == 'task' else v.to(self.local_rank,non_blocking=True))
                for k, v in batch_dict['input_dict'].items()
            }
            with autocast(enabled=self.cfg.use_amp,device_type=self.device_type):
                y_hats = self.model(**input_dict)
                y = batch_dict['pos_item_embedding'].to(self.local_rank,non_blocking=True)
                neg_items_emb_tensors = batch_dict['neg_items_embedding'].to(self.local_rank,non_blocking=True)
                neg_items_mask = batch_dict['neg_items_mask'].to(self.local_rank,non_blocking=True)
                loss = self.loss(
                    batch_y=y,
                    batch_y_hat=y_hats,
                    batch_negative_samples=neg_items_emb_tensors,
                    batch_negative_mask=neg_items_mask
                )
                original_loss = loss.clone().detach()
                loss = loss / self.cfg.accumulation_steps
            self.scaler.scale(loss).backward()
            update_grad = ((step + 1) % self.cfg.accumulation_steps == 0) or ((step + 1) == len(self.train_dataloader))
            if update_grad:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
            total_loss += original_loss
            metrics = {
                'batch_step': epoch * len(self.train_dataloader) + step,
                'learning_rate': self.scheduler.get_last_lr()[0] if self.scheduler else self.cfg.learning_rate,
                'loss/train/batch': original_loss.item()
            }
            self.log(
                level='info',
                msg=str(metrics),
                metrics=metrics
            )

        metrics = {
            'epoch':epoch,
            'loss/train/epoch': total_loss.item() / len(self.train_dataloader),
        }
        self.log(
            level='info',
            msg=str(metrics),
            metrics=metrics
        )

    @torch.no_grad()
    def valid_epoch(self, epoch):
        self.model.eval()
        valid_processor = tqdm(self.valid_dataloader, desc=f"Epoch {epoch}/{self.cfg.n_epochs}")
        total_loss = torch.tensor(0.0, device=self.local_rank, dtype=torch.float)
        top_k_list = [1,5,10,15,30,50]
        all_y_hats = []
        all_pos_item_ids = []
        for step,batch_dict in enumerate(valid_processor):
            input_dict = {
                k: (v if k == 'task' else v.to(self.local_rank,non_blocking=True))
                for k, v in batch_dict['input_dict'].items()
            }
            with autocast(enabled=self.cfg.use_amp,device_type=self.device_type):
                y_hats = self.model(**input_dict)
                y = batch_dict['pos_item_embedding'].to(self.local_rank,non_blocking=True)
                neg_items_emb_tensors = batch_dict['neg_items_embedding'].to(self.local_rank,non_blocking=True)
                neg_items_mask = batch_dict['neg_items_mask'].to(self.local_rank,non_blocking=True)
                loss = self.loss(
                    batch_y=y,
                    batch_y_hat=y_hats,
                    batch_negative_samples=neg_items_emb_tensors,
                    batch_negative_mask=neg_items_mask
                )
                original_loss = loss.clone().detach()
            total_loss += original_loss
            metrics = {
                'loss': original_loss.item(),
                # **self.compute_recall_metrics(
                #     top_k_list=top_k_list,
                #     dataloader=self.valid_dataloader,
                #     y_hats=y_hats,
                #     pos_item_ids=pos_item_['ids']
                # )
            }
            metrics = {
                'batch_step': epoch * len(self.valid_dataloader) + step,
                **{f'{k}/valid/batch':v for k,v in metrics.items()}
            }
            self.log(
                level='info',
                msg=str(metrics),
                metrics=metrics
            )
            all_y_hats.append(y_hats.clone().detach())
            all_pos_item_ids.extend(batch_dict['pos_item_id'])

        all_y_hats = torch.cat(all_y_hats,dim=0)
        metrics = {
            'loss': total_loss.item() / len(self.valid_dataloader),
        }
        if epoch==0 or (epoch+1)%5 == 0 or epoch>=150:
            metrics.update(
                self.compute_recall_metrics(
                    top_k_list=top_k_list,
                    dataloader=self.valid_dataloader,
                    y_hats=all_y_hats,
                    pos_item_ids=all_pos_item_ids
                )
            )
        self.try_save_checkpoint(metrics=metrics, epoch=epoch)
        metrics = {
            'epoch':epoch,
            **{f'{k}/valid/epoch':v for k,v in metrics.items()}
        }
        self.log(
            level='info',
            msg=str(metrics),
            metrics=metrics
        )
    def compute_recall_metrics(
        self,
        top_k_list: List[int],
        dataloader: DataLoader,
        y_hats: torch.Tensor,
        pos_item_ids: List[int]
    ):
        y_hats = y_hats.clone().detach()
        dataset = cast(PolyvoreComplementaryItemRetrievalDataset, dataloader.dataset)
        candidate_pools = dataset.candidate_pools

        category_to_queries = defaultdict(list)
        category_to_gt = defaultdict(list)

        for i,item_id in enumerate(pos_item_ids):
            c_id = dataset.metadata[item_id]['category_id']
            category_to_queries[c_id].append(y_hats[i].cpu())
            category_to_gt[c_id].append(candidate_pools[c_id]['index'][item_id])

        max_length = max(len(queries) for queries in category_to_queries.values())

        category_to_queries_padded = []
        candidate_tensors = []
        gt_padded = []
        mask = []
        for c_id,queries in category_to_queries.items():
            qs_tensor = torch.stack(queries,dim=0)
            q_length = qs_tensor.shape[0]
            category_to_queries_padded.append(
                torch.nn.functional.pad(
                    qs_tensor,
                    (0,0,0,max_length-q_length),
                    value=0
                ).to(self.local_rank)
            )
            candidate_tensors.append(
                candidate_pools[c_id]['embeddings'].clone().detach().to(self.local_rank)
            )
            gt_padded.append(
                category_to_gt[c_id]+[-1]*(max_length-q_length)
            )
            mask.append([1]*q_length+[0]*(max_length-q_length))

        category_to_queries_tensor_padded = torch.stack(category_to_queries_padded,dim=0)
        candidate_tensors = torch.stack(candidate_tensors,dim=0)
        gt_index_tensor = torch.tensor(gt_padded,dtype=torch.long,device=self.local_rank) #  after padded [C, max_len]
        mask_tensor = torch.tensor(mask,dtype=torch.bool,device=self.local_rank) # [C, max_len]

        with autocast(enabled=self.cfg.use_amp,device_type=self.device_type):
            dists = torch.cdist(category_to_queries_tensor_padded, candidate_tensors)  # [C, max_len, 3000]
        top_k_index = torch.topk(dists, k=max(top_k_list), largest=False).indices  # [C, max_len, K] k 个 index

        metrics = defaultdict(float)
        for k in top_k_list:
            hits = (mask_tensor.unsqueeze(-1)&(top_k_index[:, :, :k] == gt_index_tensor.unsqueeze(-1))).any(dim=-1).float()
            metric = f'Recall@{k}'
            metrics[metric] = hits.sum().item() / mask_tensor.sum().item()
        return metrics

    # def compute_recall_metrics(
    #     self,
    #     top_k_list: List[int],
    #     dataloader: DataLoader,
    #     y_hats: torch.Tensor,
    #     pos_item_ids: List[int],
    #     split_parts: int = 10  # 🔥 把 batch 分成几块处理
    # ):
    #     y_hats = y_hats.clone().detach()
    #     dataset = cast(PolyvoreComplementaryItemRetrievalDataset, dataloader.dataset)
    #     candidate_pools = dataset.candidate_pools
    #     metrics = {f"Recall@{k}": 0.0 for k in top_k_list}
    #     total = len(y_hats)
    #
    #     # 自动计算每块大小
    #     split_batch_size = (total + split_parts - 1) // split_parts  # 向上取整
    #
    #     for start in range(0, total, split_batch_size):
    #         end = min(start + split_batch_size, total)
    #         y_chunk = y_hats[start:end]
    #         pos_ids_chunk = pos_item_ids[start:end]
    #
    #         candidate_embeddings = []
    #         ground_true_index = []
    #
    #         for item_id in pos_ids_chunk:
    #             c_id = dataset.metadata[item_id]['category_id']
    #             pool = candidate_pools[c_id]
    #             candidate_embeddings.append(pool['embeddings'])  # [Pool_size, D]
    #             ground_true_index.append(pool['index'][item_id])
    #
    #         candidate_pool_tensor = torch.stack(candidate_embeddings, dim=0).to(self.local_rank)  # [B, Pool_size, D]
    #         query_expanded = y_chunk.unsqueeze(1)  # [B, 1, D]
    #         distances = torch.norm(candidate_pool_tensor - query_expanded, dim=-1)  # [B, Pool_size]
    #         top_k_index = torch.topk(distances, k=max(top_k_list), largest=False).indices  # [B, K]
    #
    #         ground_true_tensor = torch.tensor(ground_true_index, dtype=torch.long,device=self.local_rank)
    #
    #         for k in top_k_list:
    #             hits = (top_k_index[:, :k] == ground_true_tensor.unsqueeze(1)).any(dim=1).float().sum().item()
    #             metrics[f"Recall@{k}"] += hits
    #
    #     for k in top_k_list:
    #         metrics[f"Recall@{k}"] /= total  # 平均化结果
    #
    #     return metrics
    def try_save_checkpoint(self, metrics: Dict[str, float], epoch: int):
        if epoch<=150:
            return
        for metric,metric_value in metrics.items():
            sign = 1 if metric=='loss' else -1
            best = self.best_metrics.get(metric, sign * np.inf)
            if metric_value * sign < best * sign:
                self.best_metrics[metric] = metric_value
                ckpt_name = f"{self.model_cfg.model_name}_best_{metric}"
                self.save_checkpoint(ckpt_name=ckpt_name,epoch=epoch)
                self.log(
                    level='info',
                    msg=f"✅ New best {metric}: {metric_value:.4f}, saved as {ckpt_name}.pth"
                )
    @torch.no_grad()
    def test(self):
        self.model.eval()
        test_processor = tqdm(self.test_dataloader, desc=f"Test")
        top_k_list = [1, 5, 10, 15, 30, 50]
        all_y_hats = []
        all_pos_item_ids = []
        for step, batch_dict in enumerate(test_processor):
            input_dict = {
                k: (v if k == 'task' else v.to(self.local_rank))
                for k, v in batch_dict['input_dict'].items()
            }
            with autocast(enabled=self.cfg.use_amp, device_type=self.device_type):
                y_hats = self.model(**input_dict)
            all_y_hats.append(y_hats.clone().detach())
            all_pos_item_ids.extend(batch_dict['pos_item_id'])

        all_y_hats = torch.cat(all_y_hats, dim=0)
        metrics = self.compute_recall_metrics(
                top_k_list=top_k_list,
                dataloader=self.test_dataloader,
                y_hats=all_y_hats,
                pos_item_ids=all_pos_item_ids
            )
        metrics = {
            **{f'{k}/test': v for k, v in metrics.items()}
        }
        self.log(
            level='info',
            msg=str(metrics),
            metrics=metrics
        )

    def hook_after_setup(self):
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = cast(
            OutfitTransformer,
            self.model
        )
        if self.world_size > 1 and self.run_mode == 'test':
            raise ValueError("测试模式下不支持分布式")
        ckpt_name_prefix = self.model_cfg.model_name
        if self.run_mode == 'train-valid':
            ckpt_path = self.cfg.checkpoint_dir.parent / 'compatibility_prediction' / f'{ckpt_name_prefix}_best_AUC.pth'
        elif self.run_mode == 'test':
            ckpt_path = self.cfg.checkpoint_dir / f'{ckpt_name_prefix}_best_Recall@1.pth'
        else:
            raise ValueError("未知的运行模式")
        self.load_checkpoint(ckpt_path=ckpt_path, only_load_model=True)

        self.cfg = cast(CIRTrainConfig, self.cfg)
        self.loss = cast(
            SetWiseRankingLoss,
            self.loss
        )

    def load_model(self) -> nn.Module:
        return OutfitTransformer(cfg=self.model_cfg)

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

    def setup_train_and_valid_dataloader(self, sample_mode: Literal['easy','hard']='easy'):
        self.sample_mode = sample_mode
        prefix = f"{self.model_cfg.model_name}_{PolyvoreItemDataset.embed_file_prefix}"
        item_embeddings = self.load_embeddings(embed_file_prefix=prefix)
        self.setup_train_dataloader(negative_sample_mode=sample_mode, item_embeddings=item_embeddings)
        self.setup_valid_dataloader(negative_sample_mode=sample_mode, item_embeddings=item_embeddings)

    def setup_train_dataloader(self, negative_sample_mode: Literal['easy', 'hard'], item_embeddings=None):
        train_dataset = PolyvoreComplementaryItemRetrievalDataset(
            polyvore_type=self.cfg.polyvore_type,
            mode='train',
            embedding_dict=item_embeddings,
            load_image=self.cfg.load_image,
            dataset_dir=self.cfg.dataset_dir,
            negative_sample_mode=negative_sample_mode
        )
        sampler = None
        self.train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            sampler=sampler,
            num_workers=self.cfg.dataloader_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.train_processor
        )

    def setup_valid_dataloader(self, negative_sample_mode: Literal['easy', 'hard'], item_embeddings=None):

        valid_dataset = PolyvoreComplementaryItemRetrievalDataset(
            polyvore_type=self.cfg.polyvore_type,
            mode='valid',
            embedding_dict=item_embeddings,
            load_image=self.cfg.load_image,
            dataset_dir=self.cfg.dataset_dir,
            negative_sample_mode=negative_sample_mode
        )
        sampler = None
        self.valid_dataloader = DataLoader(
            dataset=valid_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.cfg.dataloader_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.valid_processor
        )

    def setup_test_dataloader(self):
        prefix = f"{self.model_cfg.model_name}_{PolyvoreItemDataset.embed_file_prefix}"
        item_embeddings = self.load_embeddings(embed_file_prefix=prefix)
        test_dataset = PolyvoreComplementaryItemRetrievalDataset(
            polyvore_type=self.cfg.polyvore_type,
            mode='test',
            embedding_dict=item_embeddings,
            load_image=self.cfg.load_image,
            dataset_dir=self.cfg.dataset_dir,
        )
        sampler = None
        self.test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.cfg.dataloader_workers,
            pin_memory=True,
            collate_fn=self.test_processor
        )

    def load_loss(self):
        return SetWiseRankingLoss(margin=self.cfg.margin)

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
    def custom_task(self, *args, **kwargs):
        pass
    def setup_custom_dataloader(self):
        pass
