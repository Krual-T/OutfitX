from dataclasses import dataclass, field
from typing import Literal

from .item_encoder_config import ItemEncoderConfig
from .transformer_config import TransformerConfig


@dataclass
class OutfitXConfig:
    # 定义填充策略，可选值为 'longest' 或 'max_length' ，默认为 'longest'
    padding: Literal['longest', 'max_length'] ='max_length'# 'longest'
    # 定义最大长度，默认为16
    max_length: int = 16
    # 定义是否进行截断，默认为True
    truncation: bool = True
    # 定义嵌入维度，默认为128 这个输出要和item_encoder的输出维度一致
    d_embed: int = 1024

    item_encoder: ItemEncoderConfig = field(default_factory=ItemEncoderConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)

    def __post_init__(self):
        self.d_embed = self.item_encoder.dim_per_modality * 2 # clip:512*2 resnet_hf_sentence_bert:64*2 slip:768*2
        model_name = ""
        if self.item_encoder.type == 'clip':
            model_name = self.item_encoder.clip_model_name.split('/')[-1]
        elif self.item_encoder.type =='resnet_hf_sentence_bert':
            model_name = self.item_encoder.text_model_name.split('/')[-1]
        elif self.item_encoder.type =='slip':
            model_name = self.item_encoder.slip_model_name.split('/')[-1]
        self.model_name = model_name