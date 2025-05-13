from dataclasses import dataclass
from typing import Literal


@dataclass
class ItemEncoderConfig:

    type: Literal['clip', 'resnet_hf_sentence_bert'] = 'clip'

    norm_out: bool = True
    # 定义聚合方法，可选值为 'concat' 、 'sum' 或 'mean'，默认为 'concat'
    aggregation_method: Literal['concat', 'sum', 'mean'] = 'concat'

    def __post_init__(self):
        # 根据不同的类型配置不同的encoder
        if self.type == 'clip':
            self.clip_model_name: str = "Marqo/marqo-fashionCLIP"# "Marqo/marqo-fashionSigLIP"#"patrickjohncyh/fashion-clip"
            dim_per_modality_embed = 512
        elif self.type == 'resnet_hf_sentence_bert':
            self.text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
            dim_per_modality_embed = 64
        self.dim_per_modality:int = dim_per_modality_embed