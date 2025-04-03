from dataclasses import dataclass
from typing import Literal

@dataclass
class OutfitTransformerConfig:
    # 定义填充策略，可选值为'longest'或'max_length'，默认为'longest'
    padding: Literal['longest', 'max_length'] = 'longest'
    # 定义最大长度，默认为16
    max_length: int = 16
    # 定义是否进行截断，默认为True
    truncation: bool = True

    # 定义物品编码文本模型的名称，默认为"sentence-transformers/all-MiniLM-L6-v2"
    item_enc_text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # 定义每个模态的物品编码维度，默认为128
    item_enc_dim_per_modality: int = 128
    # 定义是否对物品编码进行归一化输出，默认为True
    item_enc_norm_out: bool = True
    # 定义聚合方法，可选值为'concat'、'sum'或'mean'，默认为'concat'
    aggregation_method: Literal['concat', 'sum', 'mean'] = 'concat'

    # 定义Transformer的头数，默认为16
    transformer_n_head: int = 16 # Original: 16
    # 定义Transformer的前馈网络维度，默认为2024
    transformer_d_ffn: int = 2024 # Original: Unknown
    # 定义Transformer的层数，默认为6
    transformer_n_layers: int = 6 # Original: 6
    # 定义Transformer的dropout率，默认为0.3
    transformer_dropout: float = 0.3 # Original: Unknown
    # 定义是否对Transformer的输出进行归一化，默认为False
    transformer_norm_out: bool = False

    # 定义嵌入维度，默认为128
    d_embed: int = 128