import pathlib
import json
import random
from collections import Counter, defaultdict

from typing import Literal, List, cast
from unittest import TestCase

import pandas as pd
import torch

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
        self.polyvore_type = polyvore_type
        self.mode = mode
        self.large_category_threshold = 0 if mode == 'train' else 3000
        self.negative_sample_fine_grained = 'semantic_category' if negative_sample_mode == 'easy' else 'category_id'
        self.negative_sample_k = negative_sample_k

        self.large_categories = self.__get_large_categories()
        self.cir_dataset = self.__load_split_dataset()
        self.negative_pool = self.__build_negative_pool()
        self.candidate_pools = self.__build_candidate_pool() if self.mode != 'train' else {}


    def __len__(self):
        return len(self.cir_dataset)

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
        # 获取 negative_items_embedding
        negative_items_embedding = [
            self.embedding_dict[item_id] for item_id in negative_item_ids
        ]
        return query, negative_items_embedding

    def __load_split_dataset(self) -> List[dict]:
        path = self.dataset_dir / self.polyvore_type / f'{self.mode}.json'
        with open(path, 'r',encoding='utf-8') as f:
            raw_data = json.load(f)
        result = []
        for outfit in raw_data:
            item_ids = outfit["item_ids"]
            pos_idx_list = [
                i for i, item_id in enumerate(item_ids)
                if self.metadata[item_id]["category_id"] in self.large_categories
            ]
            if pos_idx_list:
                result.append({
                    "item_ids": item_ids,
                    "positive_idx_list": pos_idx_list
                })
        return result

    def __get_large_categories(self) -> set:
        counts = Counter(
            item["category_id"] for item in self.metadata.values() if "category_id" in item
        )
        return {cid for cid, count in counts.items() if count >= self.large_category_threshold}


    def __build_negative_pool(self):
        negative_pool = defaultdict(list)
        for item in self.metadata.values():
            sample_key = item[self.negative_sample_fine_grained]
            negative_pool[sample_key].append(item['item_id'])
        return negative_pool

    def __get_negative_sample(self, item_id) -> List[int]:
        k = self.negative_sample_k
        item_meta = self.metadata[item_id]
        sample_key = item_meta[self.negative_sample_fine_grained]
        pool = self.negative_pool.get(sample_key, [])
        filtered = [x for x in pool if x != item_id]
        if len(filtered) < k:
            print(f"⚠️ 类别 {self.negative_sample_fine_grained} 负样本不足 {k} 个，仅有 {len(filtered)} 个")
        return random.sample(filtered, k) if len(filtered) >= k else filtered

    def __build_candidate_pool(self) -> dict:
        candidate_max_size = 3000
        candidate_pool = {}
        # set item_id集合
        split_item_ids = {iid for sample in self.cir_dataset for iid in sample["item_ids"]}
        category_to_all = defaultdict(list)
        category_to_split = defaultdict(set)

        for item_id, item in self.metadata.items():
            cid = item.get("category_id")
            if cid in self.large_categories:
                category_to_all[cid].append(item_id)
                if item_id in split_item_ids:
                    category_to_split[cid].add(item_id)

        for cid in self.large_categories:
            used = list(category_to_split[cid])
            replenish = list(set(category_to_all[cid]) - set(used))
            random.shuffle(replenish)
            total = used + replenish[:max(0, candidate_max_size - len(used))]
            total = total[:candidate_max_size]
            random.shuffle(total)

            index_map = {item_id: idx for idx, item_id in enumerate(total)}

            # ✅ embedding tensor
            try:
                embeddings = torch.stack([
                    torch.tensor(self.embedding_dict[item_id],dtype=torch.float)
                    for item_id in total
                ])
            except KeyError as e:
                print(f"⚠️ embedding_dict 缺失 item_id: {e}")
                raise e

            candidate_pool[cid] = {
                'item_ids': total,
                'index': index_map,
                'embeddings': embeddings  # shape: [3000, D]
            }

        print(f"✅ 候选池构建完毕：每类 {candidate_max_size} 个")
        return candidate_pool

    @staticmethod
    def train_collate_fn(batch):
        """
        弃用，因为在processor中已经处理了
        :param batch:
        :return:
        """
        query_iter, neg_items_emb_iter = zip(*batch)
        queries = [query for query in query_iter]
        pos_item_embeddings = torch.stack([
            torch.tensor(
                query.target_item.embedding,
                dtype=torch.float,
            )
            for query in queries
        ])
        neg_items_emb_tensors = torch.stack([
            torch.stack([
                torch.tensor(
                    item_emb,
                    dtype=torch.float,
                )
                for item_emb in neg_items_emb
            ])
            for neg_items_emb in neg_items_emb_iter
        ])

        return queries, pos_item_embeddings, neg_items_emb_tensors
    @staticmethod
    def valid_collate_fn(batch):
        """
        弃用，因为在processor中已经处理了
        :param batch:
        :return:
        """
        query_iter, neg_items_emb_iter = zip(*batch)
        queries = [query for query in query_iter]
        pos_item_ = {
            'ids': [
                query.target_item.item_id for query in queries
            ],
            'embeddings': torch.stack([
                torch.tensor(
                    query.target_item.embedding,
                    dtype=torch.float
                )
                for query in queries
            ])
        }
        neg_items_emb_tensors = torch.stack([
            torch.stack([
                torch.tensor(
                    item_emb,
                    dtype=torch.float
                )
                for item_emb in neg_items_emb
            ])
            for neg_items_emb in neg_items_emb_iter
        ])
        return queries, pos_item_, neg_items_emb_tensors
    @staticmethod
    def test_collate_fn(batch):
        """
        弃用，因为在processor中已经处理了
        :param batch:
        :return:
        """
        query_iter, _ = zip(*batch)
        queries = [query for query in query_iter]
        pos_item_ids = [query.target_item.item_id for query in queries]
        return queries, pos_item_ids

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

    def test_build_candidate_pool(self):
        """
        根据大类构建候选池：
        - 每个大类包含 3000 个 item_ids
        - 来自 valid.json 中已出现的 item_ids + metadata 中补充
        """
        dataset_dir = ROOT_DIR / 'datasets' / 'polyvore'
        metadata_path = dataset_dir / "item_metadata.json"
        valid_path = dataset_dir / "nondisjoint" / "valid.json"
        CATEGORY_KEY = "category_id"
        THRESHOLD = 3000
        TARGET_SIZE = 3000

        # ✅ 加载 metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)
            metadata = {item["item_id"]: item for item in raw_list}

        # ✅ 构建 category -> item_ids 映射
        category_to_all_ids = defaultdict(list)
        for item_id, item in metadata.items():
            cid = item[CATEGORY_KEY]
            category_to_all_ids[cid].append(item_id)

        # ✅ 统计每个类别数量，选出大类
        category_counts = Counter(item[CATEGORY_KEY] for item in metadata.values())
        large_categories = {cat for cat, count in category_counts.items() if count >= THRESHOLD}

        # ✅ 加载 valid 中出现的 item_ids
        with open(valid_path, "r", encoding="utf-8") as f:
            valid_outfits = json.load(f)
        valid_item_ids = {item_id for outfit in valid_outfits for item_id in outfit["item_ids"]}

        # ✅ 分类 valid 中的 item_ids 到各大类
        category_to_valid_ids = defaultdict(set)
        for item_id in valid_item_ids:
            cid = metadata.get(item_id, {}).get(CATEGORY_KEY)
            if cid in large_categories:
                category_to_valid_ids[cid].add(item_id)

        # ✅ 构建候选池
        candidate_pool = dict()
        for cid in large_categories:
            valid_ids = list(category_to_valid_ids.get(cid, set()))
            extra_ids = list(set(category_to_all_ids[cid]) - set(valid_ids))
            random.shuffle(extra_ids)  # 随机补齐
            total_ids = valid_ids + extra_ids[:max(0, TARGET_SIZE - len(valid_ids))]
            if len(total_ids) < TARGET_SIZE:
                print(f"⚠️ 类别 {cid} 无法凑满 {TARGET_SIZE} 个（仅 {len(total_ids)} 个）")
            candidate_pool[cid] = total_ids[:TARGET_SIZE]  # 精确截断

        # ✅ 输出统计
        print("\n📦 候选池构建完毕（每类 3000 个 item_ids）")
        for cid, items in candidate_pool.items():
            print(f"类别 {cid}: {len(items)} items")

        # # ✅ 如有需要，可写入文件
        # # with open("candidate_pools.json", "w", encoding="utf-8") as f:
        # #     json.dump(candidate_pools, f, ensure_ascii=False, indent=2)
        #
        # return candidate_pools

class TestValidDataset(TestCase):
    def test_valid_dataset(self):
        """
        问题：valid中的大类全部小于3000??
        结论：yes
        """
        dataset_dir = ROOT_DIR / 'datasets' / 'polyvore'
        metadata_path = dataset_dir / "item_metadata.json"
        test_path = dataset_dir / "nondisjoint" / "valid.json"
        CATEGORY_KEY = "category_id"
        THRESHOLD = 3000

        # ✅加载 metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)
            metadata = {item["item_id"]: item for item in raw_list}

        # ✅加载 test outfits
        with open(test_path, "r", encoding="utf-8") as f:
            test_outfits = json.load(f)

        # ✅统计每个类别的全局数量，选出大类
        category_counts = Counter(item[CATEGORY_KEY] for item in metadata.values())
        large_categories = {cat for cat, count in category_counts.items() if count >= THRESHOLD}

        # ✅统计 test 中大类的数量
        test_item_ids = {item_id for outfit in test_outfits for item_id in outfit["item_ids"]}
        test_category_counter = Counter()

        for item_id in test_item_ids:
            cid = metadata.get(item_id, {}).get(CATEGORY_KEY)
            if cid in large_categories:
                test_category_counter[cid] += 1

        # ✅输出
        print(f"\n📊 Valid 中大类分布（超过 3000？）")
        print(f"{'Category ID':>12s} | {'Test Count':>10s} | {'Needs Fill':>10s}")
        print("-" * 40)
        for cid in sorted(large_categories, key=lambda x: -test_category_counter[x]):
            count = test_category_counter[cid]
            need_fill = "❌ No" if count >= THRESHOLD else "✅ Yes"
            print(f"{cid:>12} | {count:>10} | {need_fill:>10}")

    def test_outfit_contains_large_category(self):
        """
        测试 valid.json 中每个 outfit 的 item_ids 是否包含大类
        输出包含大类的个数和完全不包含大类的个数
        """
        dataset_dir = ROOT_DIR / 'datasets' / 'polyvore'
        metadata_path = dataset_dir / "item_metadata.json"
        valid_path = dataset_dir / "nondisjoint" / "valid.json"
        CATEGORY_KEY = "category_id"
        THRESHOLD = 3000

        # ✅加载 metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)
            metadata = {item["item_id"]: item for item in raw_list}

        # ✅加载 valid outfits
        with open(valid_path, "r", encoding="utf-8") as f:
            valid_outfits = json.load(f)

        # ✅统计每个类别的全局数量，选出大类
        category_counts = Counter(item[CATEGORY_KEY] for item in metadata.values())
        large_categories = {cat for cat, count in category_counts.items() if count >= THRESHOLD}

        # ✅统计每个 outfit 是否包含大类
        contains_large_count = 0
        not_contains_large_count = 0

        for outfit in valid_outfits:
            item_ids = outfit.get("item_ids", [])
            has_large = False
            for item_id in item_ids:
                category_id = metadata.get(item_id, {}).get(CATEGORY_KEY)
                if category_id in large_categories:
                    has_large = True
                    break
            if has_large:
                contains_large_count += 1
            else:
                not_contains_large_count += 1

        # ✅输出
        print("\n📊 Outfit 中是否包含大类统计")
        print(f"✅ 至少包含一个大类的 outfit 数量：{contains_large_count}")
        print(f"❌ 完全不包含大类的 outfit 数量：{not_contains_large_count}")

    def test_valid_covers_all_large_categories(self):
        """
        检查 valid.json 是否覆盖了所有的大类（至少包含一个 item）
        """
        dataset_dir = ROOT_DIR / 'datasets' / 'polyvore'
        metadata_path = dataset_dir / "item_metadata.json"
        valid_path = dataset_dir / "nondisjoint" / "valid.json"
        CATEGORY_KEY = "category_id"
        THRESHOLD = 3000

        # ✅ 加载 metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)
            metadata = {item["item_id"]: item for item in raw_list}

        # ✅ 统计大类
        category_counts = Counter(item[CATEGORY_KEY] for item in metadata.values())
        large_categories = {cat for cat, count in category_counts.items() if count >= THRESHOLD}

        # ✅ 加载 valid item_ids
        with open(valid_path, "r", encoding="utf-8") as f:
            valid_outfits = json.load(f)
        valid_item_ids = {item_id for outfit in valid_outfits for item_id in outfit["item_ids"]}

        # ✅ 提取 valid 中实际出现的类别（限大类）
        valid_categories = set()
        for item_id in valid_item_ids:
            cid = metadata.get(item_id, {}).get(CATEGORY_KEY)
            if cid in large_categories:
                valid_categories.add(cid)

        # ✅ 比较差集
        uncovered_categories = large_categories - valid_categories

        # ✅ 输出
        print(f"\n📊 大类总数：{len(large_categories)}")
        print(f"✅ valid 中出现的大类种类数：{len(valid_categories)}")
        if uncovered_categories:
            print(f"❌ 以下大类没有在 valid 中出现：{sorted(uncovered_categories)}")
        else:
            print("🎉 valid 中覆盖了全部大类！")

class TestTestDataset(TestCase):
    def test_test_dataset(self):
        """
        问题：test中的大类全部小于3000??
        结论：yes
        """
        dataset_dir = ROOT_DIR / 'datasets' / 'polyvore'
        metadata_path = dataset_dir / "item_metadata.json"
        test_path = dataset_dir / "nondisjoint" / "test.json"
        CATEGORY_KEY = "category_id"
        THRESHOLD = 3000

        # ✅加载 metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)
            metadata = {item["item_id"]: item for item in raw_list}

        # ✅加载 test outfits
        with open(test_path, "r", encoding="utf-8") as f:
            test_outfits = json.load(f)

        # ✅统计每个类别的全局数量，选出大类
        category_counts = Counter(item[CATEGORY_KEY] for item in metadata.values())
        large_categories = {cat for cat, count in category_counts.items() if count >= THRESHOLD}

        # ✅统计 test 中大类的数量
        test_item_ids = {item_id for outfit in test_outfits for item_id in outfit["item_ids"]}
        test_category_counter = Counter()

        for item_id in test_item_ids:
            cid = metadata.get(item_id, {}).get(CATEGORY_KEY)
            if cid in large_categories:
                test_category_counter[cid] += 1

        # ✅输出
        print(f"\n📊 Test 中大类分布（超过 3000？）")
        print(f"{'Category ID':>12s} | {'Test Count':>10s} | {'Needs Fill':>10s}")
        print("-" * 40)
        for cid in sorted(large_categories, key=lambda x: -test_category_counter[x]):
            count = test_category_counter[cid]
            need_fill = "❌ No" if count >= THRESHOLD else "✅ Yes"
            print(f"{cid:>12} | {count:>10} | {need_fill:>10}")

    def test_test_covers_all_large_categories(self):
        """
        检查 test.json 是否覆盖了所有的大类（至少包含一个 item）
        """
        dataset_dir = ROOT_DIR / 'datasets' / 'polyvore'
        metadata_path = dataset_dir / "item_metadata.json"
        test_path = dataset_dir / "nondisjoint" / "test.json"
        CATEGORY_KEY = "category_id"
        THRESHOLD = 3000

        # ✅ 加载 metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)
            metadata = {item["item_id"]: item for item in raw_list}

        # ✅ 统计大类
        category_counts = Counter(item[CATEGORY_KEY] for item in metadata.values())
        large_categories = {cat for cat, count in category_counts.items() if count >= THRESHOLD}

        # ✅ 加载 test item_ids
        with open(test_path, "r", encoding="utf-8") as f:
            test_outfits = json.load(f)
        test_item_ids = {item_id for outfit in test_outfits for item_id in outfit["item_ids"]}

        # ✅ 提取 test 中实际出现的类别（限大类）
        test_categories = set()
        for item_id in test_item_ids:
            cid = metadata.get(item_id, {}).get(CATEGORY_KEY)
            if cid in large_categories:
                test_categories.add(cid)

        # ✅ 比较差集
        uncovered_categories = large_categories - test_categories

        # ✅ 输出
        print(f"\n📊 大类总数：{len(large_categories)}")
        print(f"✅ test 中出现的大类种类数：{len(test_categories)}")
        if uncovered_categories:
            print(f"❌ 以下大类没有在 test 中出现：{sorted(uncovered_categories)}")
        else:
            print("🎉 test 中覆盖了全部大类！")

class TestTrainAndTestDataset(TestCase):
    def test_train_and_test_dataset(self):
        """
        train and test 在item_id级别是否有重合
        :return:
        """
        dataset_dir = ROOT_DIR / 'datasets' / 'polyvore'
        train_path = dataset_dir / "nondisjoint" / "train.json"
        test_path = dataset_dir / "nondisjoint" / "test.json"
        with open(train_path, 'r', encoding='utf-8') as f:
            train_outfits = json.load(f)
        with open(test_path, 'r', encoding='utf-8') as f:
            test_outfits = json.load(f)

        # 收集所有 item_id
        train_ids = {iid for outfit in train_outfits for iid in outfit['item_ids']}
        test_ids = {iid for outfit in test_outfits for iid in outfit['item_ids']}

        # 计算交集
        overlap = train_ids & test_ids

        print(f"✅ Train 集 item 数量: {len(train_ids)}")
        print(f"✅ Test  集 item 数量: {len(test_ids)}")
        print(f"🔥 Train/Test 重合 item 数量: {len(overlap)}")
        if overlap:
            print("🌟 重合示例（最多 10 个）：")
            for iid in list(overlap)[:10]:
                print("   -", iid)

    def test_train_pos_and_test_pos(self):
        def collect_pos_ids(polyvore_type: str, mode: str):
            ds = PolyvoreComplementaryItemRetrievalDataset(
                polyvore_type=polyvore_type,
                mode=mode,
                metadata=None,  # 会在父类里自动加载
                embedding_dict=None,  # 用不到 embedding_dict
                load_image=False
            )
            pos_ids = set()
            # cir_dataset 每个 entry 都有 item_ids 和 positive_idx_list
            for entry in ds.cir_dataset:
                item_ids = entry['item_ids']
                for idx in entry['positive_idx_list']:
                    pos_ids.add(item_ids[idx])
            return pos_ids

        train_pos = collect_pos_ids('disjoint', 'train')
        test_pos = collect_pos_ids('disjoint', 'test')

        overlap = train_pos & test_pos

        print(f"✅ Train 正样本数: {len(train_pos)}")
        print(f"✅ Test  正样本数: {len(test_pos)}")
        print(f"🔥 重合正样本数: {len(overlap)}")
        if overlap:
            print("🌟 示例重合 item_id（最多10个）：")
            for iid in list(overlap)[:10]:
                print(f"   - {iid}")