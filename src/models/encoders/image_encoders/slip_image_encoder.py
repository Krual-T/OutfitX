from typing import List, Dict, Any, Tuple, Union

import numpy as np
import open_clip
import torch
from PIL import Image
from torch import autocast

from src.models.encoders.base_encoders import BaseImageEncoder


class SigLIPImageEncoder(BaseImageEncoder):
    def __init__(
            self,
            model_context: Dict[str, Any],
    ):
        super().__init__()
        self.model_name_or_path = model_context.get('model_name_or_path',"hf-hub:Marqo/marqo-fashionSigLIP")

        if ('model' not in model_context) or ('preprocess_val' not in model_context):
            self.model, _, self.transforms = open_clip.create_model_and_transforms(self.model_name_or_path)
        else:
            self.model = model_context.get('model')
            self.transforms = model_context.get('preprocess_val')

    def processor(self, images: List[Union[np.ndarray, Image.Image]]):
        return torch.stack(
            [
                self.transforms(
                    Image.fromarray(image) if isinstance(image, np.ndarray) else image
                )
                for image in images
            ]
        )
    @torch.no_grad()
    def _forward(
            self,
            images: List[Union[np.ndarray, Image.Image]]
    ):
        transformed_images = self.processor(images=images).to(self.device)

        # 获取图像嵌入
        with autocast(device_type=self.device.type, dtype=torch.float16):
            image_embeddings = self.model.encode_image(transformed_images)

        # 返回图像嵌入
        return image_embeddings

    @property
    def image_size(self) -> Tuple[int, int]:
        image_size:Tuple[int, int] = self.model.visual.image_size
        if isinstance(image_size, tuple):
            return image_size
        elif isinstance(image_size, int):
            return image_size, image_size
        else:
            raise ValueError("Invalid image size")


    @property
    def d_embed(self) -> int:
        if self.model_name_or_path == "hf-hub:Marqo/marqo-fashionSigLIP":
            return 768
        else:
            raise NotImplementedError
