import torch
import torch.nn as nn
import numpy as np

from typing import List
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms

from base_image_encoder import BaseImageEncoder
from src.models.utils.model_utils import freeze_model

class Resnet18ImageEncoder(BaseImageEncoder):

    def __init__(
            self,
            d_embed: int = 64,
            size: int = 224,
            crop_size: int = 224,
            freeze: bool = False
    ):
        super().__init__()

        # Load pre-trained ResNet-18 and adjust the final layer to match d_embed
        self.d_embed = d_embed
        self.size = size
        self.crop_size = crop_size
        self.freeze = freeze

        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # 改变resnet18的最后一层输出维度 从1000->d_embed：64
        # 将分类器改为特征编码器
        self.model.fc = nn.Linear(
            in_features=self.model.fc.in_features,
            out_features=d_embed
        )
        if freeze:
            freeze_model(self.model)

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @property
    def image_size(self) -> int:
        return self.crop_size

    @property
    def d_embed(self) -> int:
        return self.d_embed

    def _forward(
            self,
            images: List[List[np.ndarray]]
    ):
        batch_size = len(images)
        images = sum(images, [])

        transformed_images = torch.stack(
            [self.transform(image) for image in images]
        ).to(self.device)
        image_embeddings = self.model(
            transformed_images
        )
        image_embeddings = image_embeddings.view(
            batch_size, -1, self.d_embed
        )

        return image_embeddings