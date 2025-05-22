import json
import pathlib
from typing import Literal

from src.models.datatypes import OutfitCompatibilityPredictionTask
from src.project_settings.info import PROJECT_DIR as ROOT_DIR
from .polyvore_item_dataset import PolyvoreItemDataset


class PolyvoreCompatibilityPredictionDataset(PolyvoreItemDataset):
    def __init__(
        self,
        polyvore_type:Literal['nondisjoint', 'disjoint'] = 'nondisjoint',
        mode:Literal['train', 'valid', 'test'] = 'train',
        dataset_dir:pathlib.Path = ROOT_DIR / 'datasets' / 'polyvore',
        metadata: dict = None,
        embedding_dict: dict = None,
        load_image: bool = False,
        load_image_tensor: bool = False,
    ):
        super().__init__(
            dataset_dir=dataset_dir,
            metadata=metadata,
            embedding_dict=embedding_dict,
            load_image=load_image,
            load_image_tensor=load_image_tensor
        )
        cp_dataset_path = dataset_dir / polyvore_type / 'compatibility' /f'{mode}.json'
        with open(cp_dataset_path, 'r',encoding='utf-8') as f:
            self.cp_dataset = json.load(f)

    def __len__(self):
        return len(self.cp_dataset)

    def __getitem__(self, index):
        label = self.cp_dataset[index]['label']
        query = OutfitCompatibilityPredictionTask(
            outfit=[
                self.get_item(item_id) for item_id in self.cp_dataset[index]['question']
            ]
        )
        return query, label
    @staticmethod
    def collate_fn(batch):
        """
        弃用，因为在processor中已经处理了
        :param batch:
        :return:
        """
        queries_iter, labels_iter = zip(*batch)
        return [query for query in queries_iter], [label for label in labels_iter]