import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List

class BaseTextEncoder(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    @abstractmethod
    def d_embed(self) -> int:
        raise NotImplementedError('The d_embed property must be implemented by subclasses.')

    @abstractmethod
    def _forward(
            self,
            texts: List[List[str]]
    ) -> torch.Tensor:
        raise NotImplementedError('The embed method must be implemented by subclasses.')

    def forward(
            self,
            texts: List[List[str]],
            normalize: bool = True,
            *args, **kwargs
    ) -> torch.Tensor:
        if len(set(len(text_seq) for text_seq in texts)) > 1:
            raise ValueError('All sequences in texts should have the same length.')

        text_embeddings = self._forward(texts, *args, **kwargs)

        if normalize:
            text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        return text_embeddings