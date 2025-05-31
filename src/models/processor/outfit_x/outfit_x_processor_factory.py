from typing import Type, Optional, Literal
from unittest import TestCase

from src.models import OutfitX
from src.models.configs import OutfitXConfig
from .outfit_x_fill_in_the_blank_task_processor import OutfitXFillInTheBlankTaskProcessor
from .outfit_x_compatibility_prediction_task_processor import OutfitXCompatibilityPredictionTaskProcessor
from .outfit_x_complementary_item_retrieval_processor import OutfitXComplementaryItemRetrievalTaskProcessor
from .outfit_x_precompute_embedding_processor import OutfitXPrecomputeEmbeddingTaskProcessor
from src.models.datatypes import (
    OutfitComplementaryItemRetrievalTask,
    OutfitCompatibilityPredictionTask,
    OutfitPrecomputeEmbeddingTask,
    OutfitFillInTheBlankTask
)
class OutfitXProcessorFactory:
    Tasks = OutfitX.Tasks
    @staticmethod
    def get_processor(task:Type[Tasks], cfg: Optional[OutfitXConfig]= None, run_mode:Optional[Literal['train', 'valid', 'test']] = None, *args, **kwargs):

        if cfg is None:
            cfg = OutfitXConfig()

        if task is OutfitCompatibilityPredictionTask:
            return OutfitXCompatibilityPredictionTaskProcessor(cfg=cfg)

        elif task is OutfitComplementaryItemRetrievalTask:
            if run_mode is None:
                raise ValueError("run_mode must be specified for OutfitComplementaryItemRetrievalTask")
            return OutfitXComplementaryItemRetrievalTaskProcessor(run_mode=run_mode, cfg=cfg)

        elif task is OutfitFillInTheBlankTask:
            return OutfitXFillInTheBlankTaskProcessor(cfg=cfg)

        elif task is OutfitPrecomputeEmbeddingTask:
            return OutfitXPrecomputeEmbeddingTaskProcessor(cfg=cfg)

class ProcessorFactoryTest(TestCase):
    """
    passed 4
    """
    def test_pickle_cp_processor(self):
        proc = OutfitXProcessorFactory.get_processor(
            task=OutfitCompatibilityPredictionTask,
        )
        import pickle

        pickle.dumps(proc)  # 应该能正常序列化
    def test_pickle_fitb_processor(self):
        proc = OutfitXProcessorFactory.get_processor(
            task=OutfitFillInTheBlankTask,
        )
        import pickle
        pickle.dumps(proc)  # 应该能正常序列化

    def test_pickle_pe_processor(self):
        proc = OutfitXProcessorFactory.get_processor(
            task=OutfitPrecomputeEmbeddingTask,
        )
        import pickle
        pickle.dumps(proc)  # 应该能正常序列化

    def test_pickle_cir_processor(self):
        proc = OutfitXProcessorFactory.get_processor(
            task=OutfitComplementaryItemRetrievalTask,
            run_mode='train',
        )
        import pickle
        pickle.dumps(proc)  # 应该能正常序列化
        proc = OutfitXProcessorFactory.get_processor(
            task=OutfitComplementaryItemRetrievalTask,
            run_mode='valid',
        )
        pickle.dumps(proc)  # 应该能正常序列化
        proc = OutfitXProcessorFactory.get_processor(
            task=OutfitComplementaryItemRetrievalTask,
            run_mode='test',
        )
        pickle.dumps(proc)  # 应该能正常序列化
