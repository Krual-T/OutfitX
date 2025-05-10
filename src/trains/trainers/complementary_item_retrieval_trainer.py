from typing import cast, Literal

import numpy as np
import torch
from torch import nn

from src.trains.trainers.distributed_trainer import DistributedTrainer


class ComplementaryItemRetrievalTrainer(DistributedTrainer):
    def __init__(self, cfg = None,run_mode: Literal['train-valid', 'test', 'custom'] = 'train-valid'):
        if cfg is None:
            cfg = None #TODO
        super().__init__(cfg=cfg, run_mode=run_mode)
        self.cfg = cast(None, cfg)

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
        pass

    def load_model(self) -> nn.Module:
        pass

    def load_optimizer(self) -> torch.optim.Optimizer:
        pass

    def load_scheduler(self):
        pass

    def load_scaler(self):
        pass

    def setup_train_and_valid_dataloader(self):
        pass

    def setup_test_dataloader(self):
        pass

    def setup_custom_dataloader(self):
        pass

    def load_loss(self):
        pass

    def custom_task(self, *args, **kwargs):
        pass
