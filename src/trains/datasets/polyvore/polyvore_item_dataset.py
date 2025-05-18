import json
import pathlib
from unittest import TestCase

from PIL import Image
from torch.utils.data import Dataset
from src.models.datatypes import FashionItem
from src.project_settings.info import PROJECT_DIR


class PolyvoreItemDataset(Dataset):
    embed_file_prefix = 'embedding_subset_'
    def __init__(
            self,
            dataset_dir: pathlib.Path,
            metadata: dict = None,
            embedding_dict: dict = None,
            load_image: bool = False
    ):
        self.dataset_dir = dataset_dir
        self.metadata = self.load_metadata() if metadata is None else metadata
        self.load_image = load_image
        self.categories = self.load_categories()
        self.embedding_dict = embedding_dict
        self.all_item_ids = list(self.metadata.keys())

    def __len__(self):
        return len(self.all_item_ids)

    def __getitem__(self, idx) -> FashionItem:
        item_id = self.all_item_ids[idx]
        return self.get_item(item_id)

    def load_metadata(self):
        metadata_path = self.dataset_dir / 'item_metadata.json'
        with open(metadata_path,mode='r',encoding='utf-8') as f:
            metadata_original = json.load(f)
        return {item['item_id']: item for item in metadata_original}

    def load_categories(self):
        categories_path = self.dataset_dir / 'categories.json'
        with open(categories_path,mode='r',encoding='utf-8') as f:
            categories = json.load(f)
        return categories

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
        category = self.categories[str(metadata_item['category_id'])]
        description = metadata_item['title'] if metadata_item['title'] else metadata_item['url_name']
        embedding = self.embedding_dict[item_id] if self.embedding_dict else None
        text_embedding = embedding[len(embedding)//2:] if embedding is not None else None
        image = None
        if self.load_image:
            image_path = self.dataset_dir / 'images' / f'{item_id}.jpg'
            image = Image.open(image_path)
        item = FashionItem(
            item_id=item_id,
            category=category,
            description=description,
            embedding=embedding,
            text_embedding=text_embedding,
            image=image,
            metadata=metadata_item
        )
        return item
class TestItemDataset(TestCase):
    def test_item_category(self):
        import json
        data_dir = PROJECT_DIR / 'datasets' / 'polyvore'
        # 1. 读取两个 JSON 文件
        with open(data_dir / 'item_metadata.json', 'r', encoding='utf-8') as f:
            list_data = json.load(f)  # 假设是 [{...,"category_id":3,...}, {...}, ...]

        with open(data_dir /'categories.json', 'r', encoding='utf-8') as f:
            dict_data = json.load(f)  # 假设是 {"3": "...", "4": "...", ...}

        # 2. 准备 key 集合（都转换成字符串对照）
        dict_keys = set(dict_data.keys())

        # 3. 遍历列表，找出缺失的 category_id
        missing = []
        for item in list_data:
            cid = str(item.get('category_id'))  # 保证和 dict_keys 同类型
            if cid not in dict_keys:
                missing.append(cid)

        # 4. 输出结果
        if missing:
            print(f"🔥 以下 category_id 在 dict.json 中不存在：{sorted(set(missing))}")
        else:
            print("✅ 所有 category_id 都在 dict.json 的 keys 里找到了！") # √
