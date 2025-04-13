import torch
import torch.nn.functional as F

from PIL import Image
from torch import nn
from configs import OutfitTransformerConfig
from encoders import ItemEncoder
from datatypes import FashionItem,OutfitComplementaryItemRetrievalTask,OutfitCompatibilityPredictionTask
from typing import List, Union
class OutfitTransformer(nn.Module):
    def __init__(self, cfg: OutfitTransformerConfig= None):
        super().__init__()
        self.cfg = cfg if cfg is not None else OutfitTransformerConfig()
        # 1 编码器设置
        ## 1.1 服装单品编码器
        self.item_encoder = ItemEncoder(
            text_model_name=self.cfg.item_enc_text_model_name,
            enc_dim_per_modality=self.cfg.item_enc_dim_per_modality,
            enc_norm_out=self.cfg.item_enc_norm_out,
            aggregation_method=self.cfg.aggregation_method
        )
        ## 1.2 全局服装编码器 global outfit representation
        transformer_encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.item_encoder.d_embed,
                nhead=self.cfg.transformer_n_head,
                dim_feedforward=self.cfg.transformer_d_ffn,
                dropout=self.cfg.transformer_dropout,
                batch_first=True,
                norm_first=True,
                activation=F.mish,
        )
        self.transformer_encoder =nn.TransformerEncoder(
            encoder_layer=transformer_encoder_layer,
            num_layers=self.cfg.transformer_n_layers,
            enable_nested_tensor=False
        )
        # ## 1.3 全局任务嵌入向量(学习全局任务的特征)
        # self.task_emb = nn.Parameter(
        #     torch.randn(self.item_encoder.d_embed // 2) * 0.02, requires_grad=True
        # )

        # 2 '兼容性预测'(Compatibility Prediction,CP)
        ## 2.1 服装令牌(学习一个全局外观表示 a learnable embedding)
        self.outfit_token = nn.Parameter(
            torch.randn(self.item_encoder.d_embed) * 0.02, requires_grad=True
        )
        ## 2.2 处理CP问题的前馈网络,输出outfit的兼容性得分 MLP
        self.cp_ffn = nn.Sequential(
            nn.Dropout(self.cfg.transformer_dropout),
            nn.Linear(self.item_encoder.d_embed, 1),
            nn.Sigmoid()
        )

        ## 3 '互补项检索'(Complementary Item Retrieval,CIR)
        ## 3.1 处理CIR问题的前馈网络,输出目标单品的嵌入向量 MLP
        self.cir_ffn = nn.Sequential(
            nn.Linear(self.item_encoder.d_embed, self.cfg.d_embed, bias=False)
        )
        ## 3.2 CIR任务的空图像嵌入向量
        self.target_item_image_emb = nn.Parameter(
            torch.randn(self.item_encoder.d_embed // 2) * 0.02, requires_grad=True
        )

        # 4 pad
        ## 4.1 用于填充文本和图片形状
        image_size = (self.item_enc.image_size, self.item_enc.image_size)
        self.image_pad = Image.new("RGB", image_size)
        self.text_pad = ''
        ## 4.2 用于填充outfit的嵌入向量
        self.pad_emb = nn.Parameter(
            torch.randn(self.item_encoder.d_embed) * 0.02, requires_grad=True
        )

        # 5 Others
        ## 5.1 任务字典,用于根据任务类型调用对应的forward
        self.task_ = {
            OutfitCompatibilityPredictionTask: self._cp_forward,
            OutfitComplementaryItemRetrievalTask: self._cir_forward,
            FashionItem: self._embed_item_forward
        }

    @property
    def device(self) -> torch.device:
        """Returns the device on which the model's parameters are stored."""
        return next(self.parameters()).device  # 最常用方式

    def forward(self,
                queries: List[Union[OutfitCompatibilityPredictionTask, OutfitComplementaryItemRetrievalTask, FashionItem]],
                *args, **kwargs
                ):
        _type = type(queries[0])
        _forward = self.task_[_type]
        return _forward(queries, *args, **kwargs)

    def _cp_forward(self,
                    cp_queries: List[OutfitCompatibilityPredictionTask],
                    use_precomputed_embedding: bool = False
                    )->torch.Tensor:

        return

    def _cir_forward(self,
                     cir_queries: List[OutfitComplementaryItemRetrievalTask],
                     use_precomputed_embedding: bool = False
                     )->torch.Tensor:

        return

    def _embed_item_forward(self,
        items: List[FashionItem],
        use_precomputed_embedding: bool = False
    )->torch.Tensor:

        return