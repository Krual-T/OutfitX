from dataclasses import dataclass
from typing import Literal


@dataclass
class ItemEncoderConfig:

    type: Literal['clip', 'resnet_hf_sentence_bert'] = 'clip'
    # 二选一
    # clip
    clip_model_name: str = "patrickjohncyh/fashion-clip"
    # resnet_hf_sentence_bert
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dim_per_modality: int = 64

    norm_out: bool = True
    # 定义聚合方法，可选值为'concat'、'sum'或'mean'，默认为'concat'
    aggregation_method: Literal['concat', 'sum', 'mean'] = 'concat'