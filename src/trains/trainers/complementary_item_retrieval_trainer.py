import pickle
from typing import cast, Literal

import numpy as np
import torch
from torch import nn, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses import SetWiseRankingLoss
from src.models import OutfitTransformer
from src.models.configs import OutfitTransformerConfig
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
        for step,(queries, pos_item_emb_tensors, neg_items_emb_tensors) in enumerate(train_processor):

            with autocast(enabled=self.cfg.use_amp,device_type=self.device_type):
                y_hats = self.model(queries)
                loss = self.loss(
                    batch_answer=pos_item_emb_tensors,
                    batch_negative_samples=neg_items_emb_tensors,
                    batch_y_hat=y_hats
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
            ckpt_path = self.cfg.checkpoint_dir / 'best_loss.pth'
        else:
            raise ValueError("未知的运行模式")
        self.load_checkpoint(ckpt_path=ckpt_path, only_load_model=True)

        self.cfg = cast(CIRTrainConfig, self.cfg)
        self.model = cast(
            OutfitTransformer,
            self.model
        )
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
        def collate_fn(batch):
            query_iter, pos_item_emb_iter, neg_items_emb_iter = zip(*batch)
            queries = [query for query in query_iter]

            pos_item_emb_tensors = torch.stack([
                torch.from_numpy(item).float() if isinstance(item, np.ndarray) else item
                for item in pos_item_emb_iter
            ])

            neg_items_emb_tensors = torch.stack([
                torch.stack([
                    torch.from_numpy(item).float() if isinstance(item, np.ndarray) else item
                    for item in neg_items
                ])
                for neg_items in neg_items_emb_iter
            ])

            return queries, pos_item_emb_tensors, neg_items_emb_tensors
        item_embeddings = self.load_embeddings(embed_file_prefix=PolyvoreItemDataset.embed_file_prefix)
        self.setup_train_dataloader(negative_sample_mode=sample_mode, collate_fn=collate_fn, item_embeddings=item_embeddings)
        self.setup_valid_dataloader(negative_sample_mode=sample_mode, collate_fn=collate_fn, item_embeddings=item_embeddings)

    def setup_train_dataloader(self, negative_sample_mode: Literal['easy', 'hard'], collate_fn=None, item_embeddings=None):

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
            collate_fn=collate_fn
        )

    def setup_valid_dataloader(self, negative_sample_mode: Literal['easy', 'hard'], collate_fn=None, item_embeddings=None):
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
            collate_fn=collate_fn
        )

    def setup_test_dataloader(self):
        # TODO 构建候选池：
        #   - 将test的全部大类按类别放入候选池
        #   - 使用train的大类对每个类别进行填充->3000
        # item_embeddings = self.load_embeddings(embed_file_prefix=PolyvoreItemDataset.embed_file_prefix)
        # test_dataset = PolyvoreComplementaryItemRetrievalDataset(
        #     polyvore_type=self.cfg.polyvore_type,
        #     mode='test',
        #     embedding_dict=item_embeddings,
        #     load_image=self.cfg.load_image,
        #     dataset_dir=self.cfg.dataset_dir,
        # )
        # sampler = None
        # self.test_dataloader = DataLoader(
        #     dataset=test_dataset,
        #     batch_size=self.cfg.batch_size,
        #     shuffle=False,
        #     sampler=sampler,
        #     num_workers=self.cfg.dataloader_workers,
        #     pin_memory=True,
        #     collate_fn=collate_fn
        # )
        pass
    def setup_custom_dataloader(self):
        pass

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
