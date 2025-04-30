import json
import pathlib
from typing import Literal
from src.trains.configs.base_train_config import ROOT_DIR
from .polyvore_item_dataset import PolyvoreItemDataset


class PolyvoreCompatibilityDataset(PolyvoreItemDataset):
    def __init__(
        self,
        polyvore_type:Literal['nondisjoint', 'disjoint'] = 'nondisjoint',
        mode:Literal['train', 'valid', 'test'] = 'train',
        dataset_dir:pathlib.Path = ROOT_DIR / 'datasets' / 'polyvore',
        metadata: dict = None,
        embedding_dict: dict = None,
        load_image: bool = False
    ):
        super().__init__(
            dataset_dir=dataset_dir,
            metadata=metadata,
            embedding_dict=embedding_dict,
            load_image=load_image
        )
        cp_dataset_path = dataset_dir / polyvore_type / 'compatibility' /f'{mode}.json'
        with open(cp_dataset_path, 'r') as f:
            self.cp_dataset = json.load(f)

    def __len__(self):
        return len(self.cp_dataset)

    def __getitem__(self, index):
        label = self.cp_dataset[index]['label']
        query = [
            self._get_item_by_id(item_id) for item_id in self.cp_dataset[index]['question']
        ]
        return query, label