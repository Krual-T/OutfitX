import os
import pickle

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import OutfitTransformer
from src.models.configs import OutfitTransformerConfig
from src.models.datatypes import OutfitPrecomputeEmbeddingTask
from src.models.processor import OutfitTransformerProcessorFactory
from src.trains.configs import PrecomputeEmbeddingConfig
from src.trains.trainers.distributed_trainer import DistributedTrainer
from src.trains.datasets import PolyvoreItemDataset

class PrecomputeEmbeddingScript(DistributedTrainer):
    """
    run mode only support custom
    """
    def __init__(self,cfg:PrecomputeEmbeddingConfig = None):
        if cfg is None:
            cfg = PrecomputeEmbeddingConfig()
        super().__init__(cfg=cfg, run_mode='custom')
        self.cfg = cfg
        self.model_cfg = OutfitTransformerConfig()
        self.item_dataloader = None
        self.processor = OutfitTransformerProcessorFactory.get_processor(
            cfg=self.model_cfg,
            task=OutfitPrecomputeEmbeddingTask
        )
    @torch.no_grad()
    def custom_task(self, *args, **kwargs):
        self.model.eval()
        all_ids, all_embeddings = [], []
        total_batches = len(self.item_dataloader)  # 获取batch总数，用于进度条
        self.log(f"[Rank {self.rank}] 开始预计算所有物品的embedding，共{total_batches}个batch。")

        with torch.no_grad(), tqdm(self.item_dataloader, total=total_batches, desc=f"Rank {self.rank} Precomputing", ncols=100) as pbar:
            for batch_idx, batch_dict in enumerate(pbar):
                embeddings = self.model(**batch_dict['input_dict']).cpu().detach().numpy() # (B,d_embed)
                all_ids.extend(batch_dict['item_id'])
                all_embeddings.append(embeddings)
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        precomputed_embedding_dir = self.cfg.precomputed_embedding_dir
        os.makedirs(precomputed_embedding_dir, exist_ok=True)
        prefix = f"{self.model_cfg.model_name}_{PolyvoreItemDataset.embed_file_prefix}"
        save_path = precomputed_embedding_dir / f'{prefix}{self.rank}.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump({'ids': all_ids, 'embeddings': all_embeddings}, f)

        self.log(f"[Rank {self.rank}] 预计算完成！共保存{len(all_ids)}个物品到 {save_path}")

    def setup_custom_dataloader(self):
        item_dataset = PolyvoreItemDataset(self.cfg.dataset_dir, load_image=True)
        # sampler = torch.utils.data.distributed.DistributedSampler(
        #     item_dataset,
        #     num_replicas=self.world_size,
        #     rank=self.rank,
        #     shuffle=False  # 预计算不需要打乱顺序
        # )
        self.item_dataloader = DataLoader(
            dataset=item_dataset,
            batch_size=self.cfg.batch_size,
            #sampler=sampler,
            num_workers=self.cfg.dataloader_workers,
            collate_fn=self.processor,
        )

    def load_model(self) -> nn.Module:
        return OutfitTransformer(cfg=self.model_cfg)

    def load_optimizer(self) -> torch.optim.Optimizer:
        pass
    def load_scheduler(self):
        pass
    def load_scaler(self):
        pass
    def load_loss(self):
        pass
    def train_epoch(self, epoch):
        pass
    def valid_epoch(self):
        pass
    def test(self):
        pass
    def setup_train_and_valid_dataloader(self):
        pass
    def setup_test_dataloader(self):
        pass
    def hook_after_setup(self):
        pass



