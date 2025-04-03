import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from abc import ABC, abstractmethod
from typing import List



class BaseImageEncoder(nn.Module, ABC):

    def __init__(self):
        super().__init__()

    def forward(
            self,
            images: List[List[np.ndarray]],
            normalize: bool = True,
            *args, **kwargs
    ) -> torch.Tensor:
        if not self._is_sequence_elements_length_consistent(images):
            raise ValueError('All sequences in images should have the same length.')

        image_embeddings = self._forward(images, *args, **kwargs)

        if normalize:
            image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)

        return image_embeddings

    @abstractmethod
    def _forward(
            self,
            images: List[List[np.ndarray]]
    ) -> torch.Tensor:
        raise NotImplementedError('这个_forward（image_embed）方法必须由子类来实现')

    @property
    def device(self) -> torch.device:
        """
        获取模型所在的设备（CPU/GPU）
        通过访问第一个参数的设备信息来代表整个模型的设备环境
        注意：假设所有模型参数都位于同一设备上（常规情况成立）
        :return: torch.device
        """
        return next(self.parameters()).device

    @property
    @abstractmethod
    def image_size(self) -> int:
        self._property_not_implemented()

    @property
    @abstractmethod
    def d_embed(self) -> int:
        self._property_not_implemented()

    def __is_sequence_elements_length_consistent(
            images: List[List[np.ndarray]]
    ) -> bool:
        """
        每个图像样本（List[np.ndarray]）要有一样的长度
        """
        return len(set(len(image_seq) for image_seq in images)) == 1

    def __property_not_implemented(self):
        """
        自动获取属性名并抛出 NotImplementedError
        :return:
        """
        import inspect
        # 获取调用该方法的属性名
        frame = inspect.currentframe().f_back
        prop_name = frame.f_code.co_name
        raise NotImplementedError(f"属性 '{prop_name}' 必须在子类中实现")

