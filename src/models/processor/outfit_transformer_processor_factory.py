from typing import Type, Optional

from src.models import OutfitTransformer
from src.models.configs import OutfitTransformerConfig
from .outfit_transformer_fill_in_the_blank_task_processor import OutfitTransformerFillInTheBlankTaskProcessor
from .outfit_transformer_compatibility_prediction_task_processor import OutfitTransformerCompatibilityPredictionTaskProcessor
from .outfit_transformer_complementary_item_retrieval_processor import OutfitTransformerComplementaryItemRetrievalTaskProcessor
from .outfit_transformer_precompute_embedding_processor import OutfitTransformerPrecomputeEmbeddingTaskProcessor
from src.models.datatypes import (
    OutfitComplementaryItemRetrievalTask,
    OutfitCompatibilityPredictionTask,
    OutfitPrecomputeEmbeddingTask,
    OutfitFillInTheBlankTask
)
class OutfitTransformerProcessorFactory:
    Tasks = OutfitTransformer.Tasks
    @staticmethod
    def get_processor(task:Type[Tasks],cfg: Optional[OutfitTransformerConfig]= None,*args,**kwargs):

        if cfg is None:
            cfg = OutfitTransformerConfig()

        if task is OutfitCompatibilityPredictionTask:
            return OutfitTransformerCompatibilityPredictionTaskProcessor(cfg=cfg)

        elif task is OutfitComplementaryItemRetrievalTask:
            return OutfitTransformerComplementaryItemRetrievalTaskProcessor(*args,cfg=cfg,**kwargs)

        elif task is OutfitFillInTheBlankTask:
            return OutfitTransformerFillInTheBlankTaskProcessor(cfg=cfg)

        elif task is OutfitPrecomputeEmbeddingTask:
            return OutfitTransformerPrecomputeEmbeddingTaskProcessor(cfg=cfg)