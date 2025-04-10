from typing import List, Union
from pydantic import BaseModel, Field
from fashion_item import FashionItem
import numpy as np
class OutfitCompatibilityQuery(BaseModel):
    outfit: List[Union[FashionItem, int]] = Field(
        default_factory=list,
        description="List of fashion items"
    )