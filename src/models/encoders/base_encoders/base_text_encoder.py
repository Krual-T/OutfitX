import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from typing import List
from src.models.utils.model_utils import flatten_seq_to_one_dim


class BaseTextEncoder(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            texts: List[List[str]],
            normalize: bool = True,
            *args, **kwargs
    ) -> torch.Tensor:
        if self.__is_sequence_elements_length_consistent(texts):
            raise ValueError('All sequences in texts should have the same length.')

        batch_size = len(texts)
        texts = flatten_seq_to_one_dim(texts)

        text_embeddings = self._forward(texts, *args, **kwargs)

        text_embeddings = text_embeddings.view(
            batch_size, -1, self.d_embed
        )

        if normalize:
            text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        return text_embeddings

    @abstractmethod
    def _forward(
            self,
            texts: List[str]
    ) -> torch.Tensor:
        raise NotImplementedError(f"内部方法 '_forward' 必须在子类中实现")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    @abstractmethod
    def d_embed(self) -> int:
        raise NotImplementedError(f"属性 'd_embed' 必须在子类中实现")

    def __is_sequence_elements_length_consistent(
            self,
            texts: List[List[str]]
    ) -> bool:
        """
        每个文本样本（List[str]）要有一样的长度
        """
        return len(set(len(text_seq) for text_seq in texts)) == 1