from typing import cast, Literal

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from src.losses import SetWiseRankingLoss
from src.models import OutfitTransformer
from src.models.configs import OutfitTransformerConfig
from src.trains.trainers.distributed_trainer import DistributedTrainer
from src.trains.configs import ComplementaryItemRetrievalTrainConfig as CIRTrainConfig

class ComplementaryItemRetrievalTrainer(DistributedTrainer):
    def __init__(self, cfg:CIRTrainConfig= None,run_mode: Literal['train-valid', 'test', 'custom'] = 'train-valid'):
        if cfg is None:
            cfg = CIRTrainConfig()
        super().__init__(cfg=cfg, run_mode=run_mode)
        self.cfg = cast(CIRTrainConfig, cfg)

        self.device_type = None
        self.best_metrics = {
            'AUC': 0.0,
            'Precision': 0.0,
            'Recall': 0.0,
            'F1': 0.0,
            'Accuracy': 0.0,
            'loss': np.inf,
        }

    def train_epoch(self, epoch):
        pass

    def valid_epoch(self, epoch):
        pass

    def test(self):
        pass

    def hook_after_setup(self):
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.world_size > 1 and self.run_mode == 'test':
            raise ValueError("测试模式下不支持分布式")
        if self.run_mode == 'train-valid':
            ckpt_path = self.cfg.checkpoint_dir.parent / 'compatibility_prediction' / 'best_AUC.pth'
        elif self.run_mode == 'test':
            ckpt_path = self.cfg.checkpoint_dir / 'best_AUC.pth'
        else:
            raise ValueError("未知的运行模式")
        self.load_checkpoint(ckpt_path=ckpt_path, only_load_model=True)

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

    def setup_train_and_valid_dataloader(self):
        pass

    def setup_test_dataloader(self):
        pass

    def setup_custom_dataloader(self):
        pass

    def load_loss(self):
        return SetWiseRankingLoss(margin=self.cfg.margin)

    def custom_task(self, *args, **kwargs):
        pass
