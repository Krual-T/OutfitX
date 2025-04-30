import json
import pathlib

from PIL import Image
from torch.utils.data import Dataset
from src.models.datatypes import FashionItem


class PolyvoreItemDataset(Dataset):

    def __init__(
            self,
            dataset_dir: str,
            metadata: dict = None,
            embedding_dict: dict = None,
            load_image: bool = False
    ):
        self.dataset_dir = dataset_dir
        self.metadata = self.load_metadata() if metadata is None else metadata
        self.load_image = load_image
        self.embedding_dict = embedding_dict
        self.all_item_ids = list(self.metadata.keys())

    def __len__(self):
        return len(self.all_item_ids)

    def __getitem__(self, idx) -> FashionItem:
        item_id = self.all_item_ids[idx]
        return self.get_item(item_id)

    @staticmethod
    def load_metadata(self):
        metadata_path = self.dataset_dir / 'item_metadata.json'
        with open(metadata_path) as f:
            metadata_original = json.load(f)
        return {item['item_id']: item for item in metadata_original}

    def get_item(self, item_id) -> FashionItem:
        """
        metadata_item:
            {
                "item_id": 211990161,
                "url_name": "neck print chiffon plus size",
                "description": "",
                "catgeories": "",
                "title": "",
                "related": "",
                "category_id": 15,
                "semantic_category": "tops"
            }
        """
        metadata_item = self.metadata[item_id]
        category = metadata_item['semantic_category']
        description = metadata_item['title'] if metadata_item['title'] else metadata_item['url_name']
        embedding = self.embedding_dict[item_id] if self.embedding_dict else None
        image = None
        if self.load_image:
            image_path = self.dataset_dir / 'images' / f'{item_id}.jpg'
            image = Image.open(image_path)
        item = FashionItem(
            item_id=item_id,
            category=category,
            description=description,
            embedding=embedding,
            image=image,
            metadata=metadata_item
        )
        return item