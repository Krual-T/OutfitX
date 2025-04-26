import os
import pickle

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import OutfitTransformer
from src.models.configs import OutfitTransformerConfig
from src.trains.configs import PrecomputeEmbeddingConfig
from src.trains.trainers.distributed_trainer import DistributedTrainer
from src.trains.datasets import PolyvoreItemDataset

class CompatibilityTrainer(DistributedTrainer):
    def load_model(self) -> nn.Module:
        pass

    def load_optimizer(self) -> torch.optim.Optimizer:
        pass

    def load_scheduler(self):
        pass

    def load_scaler(self):
        pass

    def setup_dataloaders(self):
        pass

    def loss(self):
        pass

    def train_epoch(self, epoch):
        pass

    def valid_epoch(self):
        pass

    def test_epoch(self):
        pass

    def custom_task(self, *args, **kwargs):
        pass