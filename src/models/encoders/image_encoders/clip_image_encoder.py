import torch
import numpy as np

from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from typing import List, Dict, Any

from src.models.utils.model_utils import freeze_model, flatten_seq_to_one_dim
from src.models.encoders.base_encoders import BaseImageEncoder

class CLIPImageEncoder(BaseImageEncoder):

    def __init__(
            self,
            model_name_or_path: str = 'patrickjohncyh/fashion-clip',
            freeze: bool = True
    ):
        # 初始化父类
        super().__init__()
        # 加载CLIPVisionModelWithProjection模型
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            model_name_or_path, weights_only=False
        )
        # 设置模型为评估模式
        self.model.eval()
        # 冻结模型
        if freeze:
            freeze_model(self.model)
        # 加载CLIPImageProcessor
        self.processor = CLIPImageProcessor.from_pretrained(
            model_name_or_path, do_convert_rgb=False
        )

    # 获取图像大小 224
    @property
    def image_size(self) -> int:
        return self.processor.size['shortest_edge']

    # 获取嵌入维度 512
    @property
    def d_embed(self) -> int:
        return self.model.config.projection_dim

    # 前向传播
    @torch.no_grad()
    def _forward(
            self,
            images: List[np.ndarray],
            processor_kargs: Dict[str, Any] = None
    ):
        # 设置processor参数
        # 确保返回类型是torch.tensor
        processor_kargs = processor_kargs if processor_kargs is not None else {}
        processor_kargs['return_tensors'] = 'pt'

        # 对图像进行预处理
        transformed_images = self.processor(
            images=images, **processor_kargs
        ).to(self.device)

        # 获取图像嵌入
        image_embeddings = self.model(
            **transformed_images
        ).image_embeds

        # 返回图像嵌入
        return image_embeddings