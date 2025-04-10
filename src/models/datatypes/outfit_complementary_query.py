from typing import List, Union
from pydantic import BaseModel, Field
from fashion_item import FashionItem


class OutfitComplementaryQuery(BaseModel):
    outfit: List[Union[FashionItem, int]] = Field(
        default_factory=list,
        description="List of fashion items"
    )
    category: str = Field(
        default="",
        description="Category of the target outfit"
    )
