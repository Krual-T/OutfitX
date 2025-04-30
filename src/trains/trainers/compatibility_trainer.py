import os
import pickle
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler

from src.losses import FocalLoss
from src.models import OutfitTransformer
from src.models.configs import OutfitTransformerConfig
from src.trains.configs.compatibility_train_config import CompatibilityTrainConfig
from src.trains.datasets.polyvore.polyvore_compatibility_dataset import PolyvoreCompatibilityDataset
from src.trains.trainers.distributed_trainer import DistributedTrainer

class CompatibilityTrainer(DistributedTrainer):

    def __init__(self,cfg:Optional[CompatibilityTrainConfig]=None):
        if cfg is None:
            cfg = CompatibilityTrainConfig()
        super().__init__(cfg=cfg)
        self.loss = FocalLoss()

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

    def setup_dataloaders(self):
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
            batch_size=self.cfg.batch_sz_per_gpu,
            sampler=train_sampler,
            num_workers=self.cfg.num_workers,
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
            batch_size=self.cfg.batch_sz_per_gpu,
            sampler=valid_sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

    def train_epoch(self, epoch):
        self.model.train()
        if hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(epoch)

        self.optimizer.zero_grad()

    def valid_epoch(self):
        pass

    def test_epoch(self):
        pass

    def custom_task(self, *args, **kwargs):
        pass