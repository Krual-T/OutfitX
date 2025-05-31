from typing import List

from .outfit_x_base_processor import OutfitXBaseProcessor
from src.models.datatypes import FashionItem, OutfitPrecomputeEmbeddingTask


class OutfitXPrecomputeEmbeddingTaskProcessor(OutfitXBaseProcessor):
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