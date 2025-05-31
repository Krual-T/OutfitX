from typing import Type, Optional, Literal
from unittest import TestCase

from src.models import OutfitX
from src.models.configs import OutfitXConfig
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
    Tasks = OutfitX.Tasks
    @staticmethod
    def get_processor(task:Type[Tasks], cfg: Optional[OutfitXConfig]= None, run_mode:Optional[Literal['train', 'valid', 'test']] = None, *args, **kwargs):

        if cfg is None:
            cfg = OutfitXConfig()

        if task is OutfitCompatibilityPredictionTask:
            return OutfitTransformerCompatibilityPredictionTaskProcessor(cfg=cfg)

        elif task is OutfitComplementaryItemRetrievalTask:
            if run_mode is None:
                raise ValueError("run_mode must be specified for OutfitComplementaryItemRetrievalTask")
            return OutfitTransformerComplementaryItemRetrievalTaskProcessor(run_mode=run_mode,cfg=cfg)

        elif task is OutfitFillInTheBlankTask:
            return OutfitTransformerFillInTheBlankTaskProcessor(cfg=cfg)

        elif task is OutfitPrecomputeEmbeddingTask:
            return OutfitTransformerPrecomputeEmbeddingTaskProcessor(cfg=cfg)

class ProcessorFactoryTest(TestCase):
    """
    passed 4
    """
    def test_pickle_cp_processor(self):
        proc = OutfitTransformerProcessorFactory.get_processor(
            task=OutfitCompatibilityPredictionTask,
        )
        import pickle

        pickle.dumps(proc)  # 应该能正常序列化
    def test_pickle_fitb_processor(self):
        proc = OutfitTransformerProcessorFactory.get_processor(
            task=OutfitFillInTheBlankTask,
        )
        import pickle
        pickle.dumps(proc)  # 应该能正常序列化

    def test_pickle_pe_processor(self):
        proc = OutfitTransformerProcessorFactory.get_processor(
            task=OutfitPrecomputeEmbeddingTask,
        )
        import pickle
        pickle.dumps(proc)  # 应该能正常序列化

    def test_pickle_cir_processor(self):
        proc = OutfitTransformerProcessorFactory.get_processor(
            task=OutfitComplementaryItemRetrievalTask,
            run_mode='train',
        )
        import pickle
        pickle.dumps(proc)  # 应该能正常序列化
        proc = OutfitTransformerProcessorFactory.get_processor(
            task=OutfitComplementaryItemRetrievalTask,
            run_mode='valid',
        )
        pickle.dumps(proc)  # 应该能正常序列化
        proc = OutfitTransformerProcessorFactory.get_processor(
            task=OutfitComplementaryItemRetrievalTask,
            run_mode='test',
        )
        pickle.dumps(proc)  # 应该能正常序列化
