import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from abc import ABC, abstractmethod
from typing import List, Union, Tuple
from PIL import Image
from src.models.utils.model_utils import flatten_seq_to_one_dim


class BaseImageEncoder(nn.Module, ABC):

    def __init__(self):
        super().__init__()

    def forward(
            self,
            images: List[Union[List[np.ndarray],List[Image.Image]]],
            normalize: bool = True,
            *args, **kwargs
    ) -> torch.Tensor:
        if not self.__is_sequence_elements_length_consistent(images):
            raise ValueError('All sequences in images should have the same length.')

        # 获取batch大小
        batch_size = len(images)
        # 将图像列表展平
        images = flatten_seq_to_one_dim(images)

        # if len(images)>0 and isinstance(images[0], Image.Image):
        #     images = [np.array(image) for image in images]

        image_embeddings = self._forward(images, *args, **kwargs)

        # 将图像嵌入调整为(batch_size, auto, d_embed)的形状
        image_embeddings = image_embeddings.view(
            batch_size, -1, self.d_embed
        )

        if normalize:
            image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)

        return image_embeddings

    @abstractmethod
    def _forward(
            self,
            images: List[Union[np.ndarray, Image.Image]]
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
    def image_size(self) -> Tuple[int, int]:
        raise NotImplementedError('这个image_size方法必须由子类来实现')

    @property
    @abstractmethod
    def d_embed(self) -> int:
        raise NotImplementedError('这个d_embed方法必须由子类来实现')

    def __is_sequence_elements_length_consistent(
            self,
            images: List[List[np.ndarray]]
    ) -> bool:
        """
        每个图像样本（List[np.ndarray]）要有一样的长度
        """
        return len(set(len(image_seq) for image_seq in images)) == 1


