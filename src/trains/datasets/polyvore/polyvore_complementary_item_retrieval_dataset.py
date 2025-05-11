import pathlib
import json
import random
from collections import Counter

from typing import Literal, List, cast
from unittest import TestCase

import pandas as pd

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
        if mode == 'test':
            large_category_threshold = 3000
        else:
            large_category_threshold = 0  # 训练/验证不过滤

        # ✅ 构建 category_id → count 映射
        category_counts = Counter()
        for item in self.metadata.values():
            cid = item.get("category_id")
            if cid is not None:
                category_counts[cid] += 1

        # ✅ 提前确定大类类别集合 type:set[int]
        self.large_categories = {
            cat for cat, count in category_counts.items()
            if count >= large_category_threshold
        }

        cir_dataset_path = dataset_dir / polyvore_type / f'{mode}.json'
        with open(cir_dataset_path, 'r') as f:
            raw_data = json.load(f)

        self.cir_dataset = []
        for outfit in raw_data:
            item_ids = outfit["item_ids"]
            positive_idx_list = [
                index for index, item_id in enumerate(item_ids)
                if (item_id in self.metadata) and (self.metadata[item_id]["category_id"] in self.large_categories)
            ]
            if positive_idx_list:
                self.cir_dataset.append({
                    "item_ids": item_ids,
                    "positive_idx_list": positive_idx_list
                })
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
        #获取 outfit positive negative的item_id
        item_ids = list(self.cir_dataset[index]['item_ids'])
        positive_idx_list = self.cir_dataset[index]['positive_idx_list']
        positive_idx = random.choice(positive_idx_list)
        positive_item_id = item_ids.pop(positive_idx)
        negative_item_ids = self.__get_negative_sample(positive_item_id)
        random.shuffle(item_ids)
        # 构建query
        query: OutfitComplementaryItemRetrievalTask = OutfitComplementaryItemRetrievalTask(
            outfit=[self.get_item(item_id) for item_id in item_ids],
            target_item=self.get_item(positive_item_id)
        )
        # 获取 positive_item_embedding
        positive_item_embedding = self.embedding_dict[positive_item_id]
        # 获取 negative_items_embedding
        negative_items_embedding = [
            self.embedding_dict[item_id] for item_id in negative_item_ids
        ]
        return query, positive_item_embedding, negative_items_embedding


class Test(TestCase):
    def test_check_semantic_category(self):
        import json
        from collections import Counter

        def analyze_semantic_categories(json_path):
            with open(json_path, "r", encoding='utf-8') as f:
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

    def test_test_dataset(self):
        dataset_dir = ROOT_DIR / 'datasets' / 'polyvore'
        metadata_path = dataset_dir / "item_metadata.json"
        test_path = dataset_dir / "nondisjoint" / "test.json"
        CATEGORY_KEY = "category_id"
        THRESHOLD = 3000

        # ✅加载 metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_raw = json.load(f)
            metadata = {item["item_id"]: item for item in metadata_raw}

        # ✅映射 item_id → category_id
        item_to_category = {
            item_id: item.get(CATEGORY_KEY) for item_id, item in metadata.items()
        }

        # ✅找出大类类别
        category_counts = Counter(item_to_category.values())
        large_categories = {cat for cat, count in category_counts.items() if count >= THRESHOLD}

        # ✅加载 test outfit
        with open(test_path, "r", encoding="utf-8") as f:
            test_outfits = json.load(f)

        valid, invalid = 0, 0
        invalid_examples = []

        # ✅检查每个 outfit 是否至少包含一个大类
        for outfit in test_outfits:
            item_ids = outfit["item_ids"]
            categories = [item_to_category.get(iid) for iid in item_ids]
            if any(cat in large_categories for cat in categories if cat is not None):
                valid += 1
            else:
                invalid += 1
                if len(invalid_examples) < 5:
                    invalid_examples.append({
                        "item_ids": item_ids,
                        "categories": categories
                    })

        # ✅输出结果
        print("\n📊 CIR Outfit 合法性检查（是否包含至少一个大类）")
        print(f"总 outfit 数量         : {len(test_outfits)}")
        print(f"✅ 包含大类的 outfit    : {valid}")
        print(f"❌ 全是小类的 outfit    : {invalid}")

        print("\n🧪 前 5 个全小类 outfit 示例：")
        for i, example in enumerate(invalid_examples, 1):
            print(f"\n示例 {i}:")
            for iid, cat in zip(example["item_ids"], example["categories"]):
                tag = "✅" if cat in large_categories else "❌"
                print(f" - {iid} → 类别 {cat} {tag}")