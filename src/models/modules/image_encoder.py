from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ..utils.model_utils import freeze_model, mean_pooling

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseImageEncoder(nn.Module, ABC):

    def __init__(self):
        super().__init__()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    @abstractmethod
    def image_size(self) -> int:
        raise NotImplementedError('The image_size property must be implemented by subclasses.')

    @property
    @abstractmethod
    def d_embed(self) -> int:
        raise NotImplementedError('The d_embed property must be implemented by subclasses.')

    @abstractmethod
    def _forward(
            self,
            images: List[List[np.ndarray]]
    ) -> torch.Tensor:
        raise NotImplementedError('The embed method must be implemented by subclasses.')

    def forward(
            self,
            images: List[List[np.ndarray]],
            normalize: bool = True,
            *args, **kwargs
    ) -> torch.Tensor:
        if len(set(len(image_seq) for image_seq in images)) > 1:
            raise ValueError('All sequences in images should have the same length.')

        image_embeddings = self._forward(images, *args, **kwargs)

        if normalize:
            image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)

        return image_embeddings