import numpy as np

from typing import Optional, Union

import torch
from PIL import Image
from pydantic import BaseModel, Field


class FashionItem(BaseModel):
    item_id: Optional[int] = Field(
        default=None,
        description="Unique ID of the item, mapped to `id` in the ItemLoader"
    )
    category: Optional[str] = Field(
        default="",
        description="Category of the item"
    )
    image: Optional[Union[Image.Image, torch.Tensor]] = Field(
        default=None,
        description="Image of the item"
    )
    description: Optional[str] = Field(
        default="",
        description="Description of the item"
    )
    metadata: Optional[dict] = Field(
        default_factory=dict,
        description="Additional metadata for the item"
    )
    embedding: Optional[np.ndarray] = Field(
        default=None,
        description="Embedding of the item"
    )
    text_embedding: Optional[np.ndarray] = Field(
        default=None,
        description="category embedding of the item"
    )
    class Config:
        arbitrary_types_allowed = True