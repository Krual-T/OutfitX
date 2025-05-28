from dataclasses import dataclass
from typing import Literal


@dataclass
class ItemEncoderConfig:
    # Contrastive Language-Image Pre-Training (CLIP) 对比语言-图像预训练模型
    #  Sigmoid loss for Language-Image Pre-training (SigLIP)
    type: Literal['clip', 'resnet_hf_sentence_bert', 'slip'] = 'clip'

    norm_out: bool = True
    # 定义聚合方法，可选值为 'concat' 、 'sum' 或 'mean'，默认为 'concat'
    aggregation_method: Literal['concat', 'sum', 'mean'] = 'concat'

    def __post_init__(self):
        # 根据不同的类型配置不同的encoder
        dim_per_modality_embed = 0
        if self.type == 'clip':
            self.clip_model_name: str = "patrickjohncyh/fashion-clip"# "Marqo/marqo-fashionCLIP"不支持HF的CLIP API 但支持open_clip
            dim_per_modality_embed = 512
        elif self.type == 'resnet_hf_sentence_bert':
            self.text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
            dim_per_modality_embed = 64
        elif self.type =='slip':
            self.slip_model_name: str = "hf-hub:Marqo/marqo-fashionSigLIP"
            dim_per_modality_embed = 768
        else:
            raise ValueError(f"Unsupported type: {self.type}")
        self.dim_per_modality:int = dim_per_modality_embed