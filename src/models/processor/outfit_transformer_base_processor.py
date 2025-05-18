from typing import List, Union, Any

import numpy as np
import torch
from PIL import Image
from torch import nn

from src.models.configs import OutfitTransformerConfig
from src.models.datatypes import FashionItem

# 作为collate使用
class OutfitTransformerBaseProcessor:
    def __init__(self, cfg:OutfitTransformerConfig):
        self.cfg = cfg
        self.image_pad = Image.new("RGB", (224, 224))
        self.text_pad = ''
        self.pad_emb = torch.zeros(self.cfg.item_encoder.dim_per_modality*2)

    def _to_tensor_and_padding(
        self,
        sequences: List[List[Any]]
    ):
        max_length = self._get_max_length(sequences)
        embeddings = self._pad_sequences(
            sequences=sequences,
            max_length=max_length,
            pad_value=self.pad_emb,
            return_tensor=True
        )
        item_length = lambda seq: min(len(seq), max_length)
        pad_length = lambda seq: max_length - item_length(seq)
        mask = torch.tensor(
            data=[
                [0] * item_length(sequence) + [1] * (pad_length(sequence))
                for sequence in sequences
            ],
            dtype=torch.bool
        )
        embeddings = embeddings.contiguous()
        mask = mask.contiguous()
        return embeddings,mask

    def _get_max_length(self, sequences):
        """
        max_length的length指的是每个outfit中的item数量
        :param sequences:序列
        :return:每个outfit中的item数量的最大值,即max_length
        """
        if self.cfg.padding == 'max_length':
            return self.cfg.max_length
        max_length = max(len(seq) for seq in sequences)

        return min(self.cfg.max_length, max_length) if self.cfg.truncation else max_length

    def _pad_sequences(self, sequences, pad_value, max_length,return_tensor=False):
        # if self.cfg.truncation OR self.cfg.padding == 'max_length'：
        # len(seq)可能大于max_length，需要对序列进行截断
        item_length = lambda seq: min(len(seq), max_length)
        pad_length = lambda seq: max_length - item_length(seq)
        if return_tensor:
            return torch.stack([
                torch.cat(
                    tensors=[
                        torch.tensor(
                            data=np.array(seq)[:item_length(seq)],
                            dtype=torch.float
                        ),
                        pad_value.expand(pad_length(seq), -1)
                    ],
                    dim=0
                )
                for seq in sequences
            ]) # 堆叠后的张量默认在子元素所在设备：self.device
        else:
            return [
                seq[:item_length(seq)] + [pad_value] * (pad_length(seq))
                for seq in sequences
            ]
