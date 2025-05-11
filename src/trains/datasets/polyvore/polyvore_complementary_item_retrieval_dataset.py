import pathlib
import json
import random

from typing import Literal, List, cast
from src.models.datatypes import FashionItem, OutfitComplementaryItemRetrievalTask
from .polyvore_item_dataset import PolyvoreItemDataset
from src.project_settings.info import PROJECT_DIR as ROOT_DIR


class PolyvoreComplementaryItemRetrievalDataset(PolyvoreItemDataset):
    def __init__(
        self,
        polyvore_type: Literal['nondisjoint', 'disjoint'] = 'nondisjoint',
        mode: Literal['train', 'valid', 'test'] = 'train',
        dataset_dir: pathlib.Path = ROOT_DIR / 'datasets' / 'polyvore',
        metadata: dict = None,
        embedding_dict: dict = None,
        load_image: bool = False,
        negative_sample_mode: Literal['easy', 'hard'] = 'easy',
        negative_sample_k: int = 10
    ):
        super().__init__(
            dataset_dir=dataset_dir,
            metadata=metadata,
            embedding_dict=embedding_dict,
            load_image=load_image
        )
        cir_dataset_path = dataset_dir / polyvore_type / f'{mode}.json'
        with open(cir_dataset_path, 'r') as f:
            self.cir_dataset = json.load(f)
        self.negative_sample_mode = negative_sample_mode
        self.negative_sample_k = negative_sample_k
        self.negative_pool = self.__build_negative_pool()


    def __len__(self):
        return len(self.cir_dataset)

    def __build_negative_pool(self):
        negative_pool = {}
        for item in self.metadata.values():
            fine_grained = "semantic_category" if self.negative_sample_mode == 'easy' else "category_id"
            sample_key = item[fine_grained]
            if sample_key not in negative_pool:
                negative_pool[sample_key] = []
            negative_pool[sample_key].append(item['item_id'])
        return negative_pool

    def __get_negative_sample(self, item_id) -> List[int]:
        k = self.negative_sample_k
        meta = self.metadata[item_id]
        key = meta["semantic_category"] if self.negative_sample_mode == 'easy' else meta["category_id"]
        pool = self.negative_pool.get(key)

        if not pool:
            print(f"⚠️ 类别 {key} 无负样本可采！")
            return []

        filtered = [x for x in pool if x != item_id]
        if len(filtered) < k:
            print(f"⚠️ 类别 {key} 负样本不足 {k} 个，仅有 {len(filtered)} 个")
        return random.sample(filtered, k) if len(filtered) >= k else filtered

    def __getitem__(self, index):
        item_ids = self.cir_dataset[index]['item_ids']
        items = [
            self.get_item(item_id=item_id) for item_id in item_ids
        ]
        random_idx = random.randrange(len(items))
        positive_item_id = items.pop(random_idx).item_id
        negative_item_ids = self.__get_negative_sample(positive_item_id)

        outfit = items
        random.shuffle(outfit)

        query: OutfitComplementaryItemRetrievalTask = OutfitComplementaryItemRetrievalTask(
            outfit=outfit,
            target_item=self.get_item(positive_item_id)
        )
        positive_item_embedding = self.embedding_dict[positive_item_id]
        negative_items_embedding = [
            self.embedding_dict[item_id] for item_id in negative_item_ids
        ]
        return query, positive_item_embedding, negative_items_embedding

def test_check_semantic_category():
    import json
    from collections import Counter

    def analyze_semantic_categories(json_path):
        with open(json_path, "r",encoding='utf-8') as f:
            item_metadata = json.load(f)

        semantic_categories = []
        missing_count = 0
        non_string_count = 0

        for item in item_metadata:
            category = item.get("category_id")
            if category is None:
                missing_count += 1
            elif not isinstance(category, int):
                non_string_count += 1
            else:
                semantic_categories.append(category)

        total_items = len(item_metadata)
        unique_categories = set(semantic_categories)
        category_counts = Counter(semantic_categories)

        print("🔍 分析结果：")
        print(f"总 item 数量: {total_items}")
        print(f"缺失 semantic_category 的数量: {missing_count}")
        print(f"semantic_category 非字符串的数量: {non_string_count}")
        print(f"唯一 semantic_category 类别数量: {len(unique_categories)}")
        print("🎯 所有类别如下：")
        for cat in sorted(unique_categories):
            print(f"  - {cat}")
        print("📊 类别出现频率（Top 10）：")
        least_common = category_counts.most_common()[::-1][:10]
        for cat, count in least_common:
            print(f"  {cat}: {count}")
        # for cat, count in category_counts.most_common(-10):
        #     print(f"  {cat}: {count}")

    # 使用方法：
    # analyze_semantic_categories("polyvore_item_metadata.json")

    metadata_path = ROOT_DIR / 'datasets' / 'polyvore' / 'item_metadata.json'
    analyze_semantic_categories(metadata_path)
if __name__ == '__main__':
    test_check_semantic_category()