import pickle
from typing import cast, Literal, List

import numpy as np
import torch
from torch import nn, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses import SetWiseRankingLoss
from src.models import OutfitTransformer
from src.models.configs import OutfitTransformerConfig
from src.models.datatypes import OutfitComplementaryItemRetrievalTask
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
        self.best_loss = np.inf

    def train_epoch(self, epoch):
        self.model.train()
        train_processor = tqdm(self.train_dataloader, desc=f"Epoch {epoch}/{self.cfg.n_epochs}")
        self.optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=self.local_rank, dtype=torch.float)
        for step,(queries, pos_item_embeddings, neg_items_emb_tensors) in enumerate(train_processor):
            y = pos_item_embeddings
            with autocast(enabled=self.cfg.use_amp,device_type=self.device_type):
                y_hats = self.model(queries)
                loss = self.loss(
                    batch_y=y.to(self.local_rank),
                    batch_y_hat=y_hats.to(self.local_rank),
                    batch_negative_samples=neg_items_emb_tensors.to(self.local_rank),
                )
                original_loss = loss.clone().detach()
            self.scaler.scale(loss).backward()
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
                'train/batch/loss': original_loss.item()
            }
            self.log(
                level='info',
                msg=str(metrics),
                metrics=metrics
            )
            train_processor.set_postfix(**metrics)

        metrics = {
            'epoch':epoch,
            'train/epoch/loss': total_loss.item() / len(self.train_dataloader),
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
        for step,(queries,pos_item_,neg_items_emb_tensors) in enumerate(valid_processor):
            y = pos_item_['embeddings']
            with autocast(enabled=self.cfg.use_amp,device_type=self.device_type):
                y_hats = self.model(queries)
                loss = self.loss(
                    batch_y=y.to(self.local_rank),
                    batch_y_hat=y_hats.to(self.local_rank),
                    batch_negative_samples=neg_items_emb_tensors.to(self.local_rank),
                )
                original_loss = loss.clone().detach()
            total_loss += original_loss
            metrics = {
                'loss': original_loss.item()
                **self.compute_recall_metrics(
                    top_k_list=top_k_list,
                    dataloader=self.valid_dataloader,
                    y_hats=y_hats,
                    pos_item_ids=pos_item_['ids']
                )
            }
            metrics = {
                'batch_step': epoch * len(self.valid_dataloader) + step,
                **{f'valid/batch/{k}':v for k,v in metrics.items()}
            }
            self.log(
                level='info',
                msg=str(metrics),
                metrics=metrics
            )
            all_y_hats.append(y_hats.clone().detach())
            all_pos_item_ids.extend(pos_item_['ids'])
            valid_processor.set_postfix(**metrics)

        all_y_hats = torch.cat(all_y_hats,dim=0)
        metrics = {
            'loss': total_loss.item() / len(self.valid_dataloader),
            **self.compute_recall_metrics(
                top_k_list=top_k_list,
                dataloader=self.valid_dataloader,
                y_hats=all_y_hats,
                pos_item_ids=all_pos_item_ids
            )
        }
        metrics = {
            'epoch':epoch,
            **{f'valid/epoch/{k}':v for k,v in metrics.items()}
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
        metrics = {}

        candidate_embeddings = []
        ground_true_index = []
        for item_id in pos_item_ids:
            c_id = dataset.metadata[item_id]['category_id']
            candidate_pool = candidate_pools[c_id]
            candidate_embeddings.append(candidate_pool['embeddings'].to(self.local_rank)) # [Pool_size, D]
            ground_true_index.append(candidate_pool['index'][item_id])
        candidate_pool_tensor = torch.stack(candidate_embeddings,dim=0) # [B, Pool_size, D]
        ground_true_index_tensor = torch.tensor(ground_true_index,dtype=torch.long, device=self.local_rank)

        query_expanded = y_hats.unsqueeze(1)  # [B, 1, D]
        distances = torch.norm(candidate_pool_tensor - query_expanded, dim=-1)
        top_k_index = torch.topk(distances, k=max(top_k_list), largest=False).indices  # [B, K]

        for k in top_k_list:
            hits = (top_k_index[:, :k] == ground_true_index_tensor.unsqueeze(1)).any(dim=1).float().sum().item()
            metrics[f"Recall@{k}"] = hits / y_hats.size(0)
        return metrics

    @torch.no_grad()
    def test(self):
        self.model.eval()
        test_processor = tqdm(self.test_dataloader, desc=f"Test")
        top_k_list = [1, 5, 10, 15, 30, 50]
        all_y_hats = []
        all_pos_item_ids = []
        for step, (queries, pos_item_ids) in enumerate(test_processor):
            with autocast(enabled=self.cfg.use_amp, device_type=self.device_type):
                y_hats = self.model(queries)
            all_y_hats.append(y_hats.clone().detach())
            all_pos_item_ids.extend(pos_item_ids)

        all_y_hats = torch.cat(all_y_hats, dim=0)
        metrics = self.compute_recall_metrics(
                top_k_list=top_k_list,
                dataloader=self.valid_dataloader,
                y_hats=all_y_hats,
                pos_item_ids=all_pos_item_ids
            )
        metrics = {
            **{f'test/{k}': v for k, v in metrics.items()}
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
        ckpt_name_prefix = self.model.cfg.model_name
        if self.run_mode == 'train-valid':
            ckpt_path = self.cfg.checkpoint_dir.parent / 'compatibility_prediction' / f'{ckpt_name_prefix}_best_AUC.pth'
        elif self.run_mode == 'test':
            ckpt_path = self.cfg.checkpoint_dir / f'{ckpt_name_prefix}_best_loss.pth'
        else:
            raise ValueError("未知的运行模式")
        self.load_checkpoint(ckpt_path=ckpt_path, only_load_model=True)

        self.cfg = cast(CIRTrainConfig, self.cfg)
        self.loss = cast(
            SetWiseRankingLoss,
            self.loss
        )

    def load_model(self) -> nn.Module:
        cfg = OutfitTransformerConfig()
        return OutfitTransformer(cfg=cfg)

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

    def setup_train_and_valid_dataloader(self, sample_mode: Literal['easy','hard']='easy'):
        item_embeddings = self.load_embeddings(embed_file_prefix=PolyvoreItemDataset.embed_file_prefix)
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
            collate_fn=PolyvoreComplementaryItemRetrievalDataset.train_collate_fn
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
            collate_fn=PolyvoreComplementaryItemRetrievalDataset.valid_collate_fn
        )

    def setup_test_dataloader(self):
        item_embeddings = self.load_embeddings(embed_file_prefix=PolyvoreItemDataset.embed_file_prefix)
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
            collate_fn=PolyvoreComplementaryItemRetrievalDataset.test_collate_fn
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
