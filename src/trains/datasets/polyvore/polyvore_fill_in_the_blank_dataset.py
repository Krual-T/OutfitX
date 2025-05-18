import json
import pathlib
from typing import Literal

import torch
from src.models.datatypes import OutfitComplementaryItemRetrievalTask
from src.project_settings.info import PROJECT_DIR as ROOT_DIR
from src.trains.datasets import PolyvoreItemDataset


class PolyvoreFillInTheBlankDataset(PolyvoreItemDataset):
    def __init__(
        self,
        polyvore_type: Literal['nondisjoint', 'disjoint'] = 'nondisjoint',
        mode: Literal['train', 'valid', 'test'] = 'test',
        dataset_dir: pathlib.Path = ROOT_DIR / 'datasets' / 'polyvore',
        metadata: dict = None,
        embedding_dict: dict = None,
        load_image: bool = False,
    ):
        super().__init__(
            dataset_dir=dataset_dir,
            metadata=metadata,
            embedding_dict=embedding_dict,
            load_image=load_image
        )
        fitb_dataset_path = dataset_dir / polyvore_type / 'fill_in_the_blank' /f'{mode}.json'
        with open(fitb_dataset_path, 'r',encoding='utf-8') as f:
            self.fitb_dataset = json.load(f)

    def __len__(self):
        return len(self.fitb_dataset)
    def __getitem__(self, idx):
        answer_index = self.fitb_dataset[idx]['label']
        query_item_ids = self.fitb_dataset[idx]['question']
        candidate_item_ids = self.fitb_dataset[idx]['answers']
        query = OutfitComplementaryItemRetrievalTask(
            outfit=[self.get_item(item_id) for item_id in query_item_ids],
            target_item=self.get_item(candidate_item_ids[answer_index])
        )
        candidate_item_embeddings = torch.stack([
            torch.tensor(self.embedding_dict[item_id], dtype=torch.float) for item_id in candidate_item_ids
        ])
        return query, candidate_item_embeddings, answer_index
    @staticmethod
    def collate_fn(batch):
        """
        弃用，因为在processor中已经处理了
        :param batch:
        :return:
        """
        queries_iter, candidate_item_embeddings_iter, batch_y_iter = zip(*batch)
        return (
            [query for query in queries_iter],
            torch.stack(candidate_item_embeddings_iter),
            torch.tensor(batch_y_iter, dtype=torch.long)
        ) # candidates [batch_size, 4, 768]