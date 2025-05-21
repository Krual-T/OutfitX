import torch
import torch.nn as nn
import numpy as np

from typing import List, Tuple, Union

from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from src.models.encoders.base_encoders import BaseImageEncoder
from src.utils.model_utils import freeze_model


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
        self._d_embed = d_embed
        self.size = size
        self.crop_size = crop_size
        self.freeze = freeze

        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False  # Change inplace to False
        # 改变resnet18的最后一层输出维度 从1000->d_embed：64
        # 将分类器改为特征编码器
        # fc（Fully Connected Layer）：最后的全连接层
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
        """
                transforms.Compose 是 PyTorch 中的一个类，用于将多个图像变换操作组合在一起。它接受一个列表作为参数，列表中的每个元素都是一个图像变换操作。这些操作将按照列表中的顺序依次执行。
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BICUBIC):

                transforms.Resize 用于调整图像的大小。self.size 指定了目标图像的大小。
                interpolation=transforms.InterpolationMode.BICUBIC 指定了插值方法为双三次插值（Bicubic interpolation），这是一种常用的图像缩放插值方法，能够在缩放图像时保持较好的图像质量。
                transforms.CenterCrop(self.crop_size):

                transforms.CenterCrop 用于从图像中心裁剪出一个固定大小的区域。self.crop_size 指定了裁剪区域的大小。
                这种操作常用于从大图中提取出感兴趣的区域，或者确保输入模型的图像具有固定的大小。
                transforms.ToTensor():

                transforms.ToTensor 用于将图像转换为 PyTorch 张量（Tensor）。图像的像素值会被归一化到 [0, 1] 的范围内。
                在深度学习中，模型通常需要以张量的形式接收输入数据，因此这一步是必要的。
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

                transforms.Normalize 用于对图像进行标准化处理。它接受两个参数：均值（mean）和标准差（std），分别用于对图像的每个通道进行标准化。
                这里的均值和标准差是针对 ImageNet 数据集的，它们是预训练模型（如 ResNet）在 ImageNet 上训练时使用的。标准化有助于加速模型的收敛，并提高模型的泛化能力。

                """

    @property
    def image_size(self) -> Tuple[int, int]:
        image_size = (self.crop_size, self.crop_size)
        return image_size
    @property
    def d_embed(self) -> int:
        return self._d_embed

    def _forward(
            self,
            images: List[Union[np.ndarray, Image.Image]]
    ):
        transformed_images = torch.stack(
            [
                self.transform(
                    Image.fromarray(image) if isinstance(image, np.ndarray) else image
                ) for image in images
            ]
        ).to(self.device)

        image_embeddings = self.model(
            transformed_images
        )

        return image_embeddings