from typing import List

from .outfit_transformer_base_processor import OutfitTransformerBaseProcessor
from src.models.datatypes import FashionItem, OutfitPrecomputeEmbeddingTask


class OutfitTransformerPrecomputeEmbeddingTaskProcessor(OutfitTransformerBaseProcessor):
    def __call__(self, batch:List[OutfitPrecomputeEmbeddingTask]):
        batch_dict = {
            'input_dict': {
                'task': OutfitPrecomputeEmbeddingTask,
                'images': [[task.fashion_item.image] for task in batch],
                'texts': [[task.fashion_item.category] for task in batch]
            },
            'item_id': [task.fashion_item.item_id for task in batch],
        }
        return batch_dict