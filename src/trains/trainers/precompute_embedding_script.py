import os
import pickle

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models import OutfitTransformer
from src.models.configs import OutfitTransformerConfig
from src.trains.configs import PrecomputeEmbeddingConfig
from src.trains.trainers.distributed_trainer import DistributedTrainer
from src.trains.datasets import PolyvoreItemDataset

class PrecomputeEmbeddingScript(DistributedTrainer):
    def __init__(self,cfg:PrecomputeEmbeddingConfig = PrecomputeEmbeddingConfig()):
        super().__init__(cfg=cfg)
        self.cfg = cfg
        self.item_dataloader = None

    def setup_dataloaders(self):
        item_dataset = PolyvoreItemDataset(self.cfg.dataset_dir,load_image=True)
        sampler = torch.utils.data.distributed.DistributedSampler(
            item_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False  # 预计算不需要打乱顺序
        )
        collate_fn = lambda batch:[item for item in batch]
        self.item_dataloader = DataLoader(
            dataset=item_dataset,
            batch_size=self.cfg.batch_sz_per_gpu,
            sampler=sampler,
            num_workers=self.cfg.n_workers_per_gpu,
            collate_fn=collate_fn
        )


    def load_model(self) -> nn.Module:
        cfg = OutfitTransformerConfig()
        return OutfitTransformer(cfg=cfg)

    def load_optimizer(self) -> torch.optim.Optimizer:
        pass

    def load_scheduler(self):
        pass

    def load_scaler(self):
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
        self.model.eval()
        all_ids,all_embeddings = [],[]
        with torch.no_grad():
            for batch in self.item_dataloader:
                embeddings = self.model.module.precompute_embeddings(batch)
                all_ids.extend([item.item_id for item in batch])
                all_embeddings.append(embeddings)
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        precomputed_embedding_dir = self.cfg.precomputed_embedding_dir
        os.makedirs(precomputed_embedding_dir,exist_ok=True)
        with open(precomputed_embedding_dir/f'embedding_subset_{self.rank}.pkl','wb') as f:
            pickle.dump({'ids': all_ids, 'embeddings': all_embeddings}, f)

if __name__ == '__main__':
    with PrecomputeEmbeddingScript() as PES:
        PES.run()
