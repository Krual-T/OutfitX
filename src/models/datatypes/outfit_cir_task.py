from typing import List, Union
from pydantic import BaseModel, Field
from fashion_item import FashionItem


class OutfitComplementaryItemRetrievalTask(BaseModel):
    outfit: List[FashionItem] = Field(
        default_factory=list,
        description="List of fashion items"
    )
    description: str = Field(
        default="",
        description="Category of the target outfit"
    )
