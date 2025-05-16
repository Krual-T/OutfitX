import pickle
from typing import Optional, Literal, cast, Union

import numpy as np
import torch
from torch import nn, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import OutfitTransformer
from src.models.configs import OutfitTransformerConfig
from src.trains.configs.fill_in_the_blank_train_config import FillInTheBlankTrainConfig
from src.trains.datasets import PolyvoreItemDataset
from src.trains.datasets.polyvore.polyvore_fill_in_the_blank_dataset import PolyvoreFillInTheBlankDataset
from src.trains.trainers.distributed_trainer import DistributedTrainer


class FillInTheBlankTrainer(DistributedTrainer):
    def __init__(self, cfg:Optional[FillInTheBlankTrainConfig]=None, run_mode:Literal['train-valid', 'test', 'custom']= 'train-valid'):
        if cfg is None:
            cfg = FillInTheBlankTrainConfig()
        super().__init__(cfg=cfg, run_mode=run_mode)
        self.cfg = cast(FillInTheBlankTrainConfig, cfg)
        self.device_type = None
        self.model_cfg = OutfitTransformerConfig()
    @torch.no_grad()
    def test(self):
        self.model.eval()

        test_dataloader_process = tqdm(self.test_dataloader, desc=f'[Test] Fill in the Blank')
        total = 0
        correct = 0
        for step, (queries, candidate_item_embeddings,y_index) in enumerate(test_dataloader_process):
            with autocast(device_type=self.device_type,enabled=self.cfg.use_amp):
                y_hats_embedding = self.model(queries) # [B,D]
                candidate_item_embeddings = candidate_item_embeddings.to(self.local_rank)  # [B,4,D]
                dists = torch.cdist(y_hats_embedding, candidate_item_embeddings, p=2)
            y_hats_index = torch.argmin(dists, dim=-1)
            y_hats_index = y_hats_index.to(self.local_rank)
            total += y_hats_index.size(0)
            correct += (y_hats_index == y_index).sum().item()
        metrics = {
            'Accuracy': float(correct / total)
        }
        self.log(
            level='info',
            msg=str(metrics),
            metrics=metrics
        )





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
            ckpt_path = self.cfg.checkpoint_dir.parent / 'complementary_item_retrieval' /f'{ckpt_name_prefix}_best_Recall@1.pth'
        else:
            raise ValueError("未知的运行模式")
        self.load_checkpoint(ckpt_path=ckpt_path, only_load_model=True)


    def load_model(self) -> nn.Module:
        return OutfitTransformer(cfg=self.model_cfg)




    def setup_test_dataloader(self):
        prefix = f"{self.model_cfg.model_name}_{PolyvoreItemDataset.embed_file_prefix}"
        item_embeddings = self.load_embeddings(embed_file_prefix=prefix)
        test_dataset = PolyvoreFillInTheBlankDataset(
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
            collate_fn=PolyvoreFillInTheBlankDataset.collate_fn
        )

    def setup_custom_dataloader(self):
        pass

    def load_loss(self):
        pass

    def train_epoch(self, epoch):
        pass

    def valid_epoch(self, epoch):
        pass

    def load_optimizer(self) -> torch.optim.Optimizer:
        pass

    def load_scheduler(self):
        pass

    def load_scaler(self):
        pass

    def setup_train_and_valid_dataloader(self):
        pass
    def custom_task(self, *args, **kwargs):
        pass

