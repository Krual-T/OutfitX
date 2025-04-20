import torch.nn.functional as F

from dataclasses import dataclass
from typing import Union, Callable
from torch import Tensor

@dataclass
class TransformerConfig:
    # 定义Transformer的头数，默认为16
    n_head: int = 16 # Original: 16
    # 定义Transformer的前馈网络维度，默认为2024
    d_ffn: int = 2024 # Original: Unknown
    # 定义Transformer的层数，默认为6
    n_layers: int = 6 # Original: 6
    # 定义Transformer的dropout率，默认为0.3
    dropout: float = 0.3 # Original: Unknown
    # 定义是否对Transformer的输出进行归一化，默认为False
    norm_out: bool = False

    batch_first:bool = True,
    norm_first:bool = True,
    activation: Union[str, Callable[[Tensor], Tensor]] = F.mish

    enable_nested_tensor: bool = False