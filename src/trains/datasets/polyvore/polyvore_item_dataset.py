import json

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
        metadata_path = dataset_dir / 'item_metadata.json'
        with open(dataset_dir) as f:
            metadata_ = json.load(f)
        self.metadata = {item['item_id']: item for item in metadata_}
        self.load_image = load_image
        self.embedding_dict = embedding_dict
        self.all_item_ids = list(self.metadata.keys())

    def __len__(self):
        return len(self.all_item_ids)

    def __getitem__(self, idx) -> FashionItem:
        item_id = self.all_item_ids[idx]
        return self.get_item(item_id)

    def get_item(self, item_id) -> FashionItem:
        metadata_item = self.metadata[item_id]
        category = metadata_item['category']
        description = metadata_item['description']
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