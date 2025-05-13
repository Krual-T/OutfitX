from dataclasses import dataclass, field
from typing import Literal

from .item_encoder_config import ItemEncoderConfig
from .transformer_config import TransformerConfig


@dataclass
class OutfitTransformerConfig:
    # 定义填充策略，可选值为 'longest' 或 'max_length' ，默认为 'longest'
    padding: Literal['longest', 'max_length'] = 'longest'
    # 定义最大长度，默认为16
    max_length: int = 16
    # 定义是否进行截断，默认为True
    truncation: bool = True
    # 定义嵌入维度，默认为128 这个输出要和item_encoder的输出维度一致
    d_embed: int = 1024

    item_encoder: ItemEncoderConfig = field(default_factory=ItemEncoderConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)