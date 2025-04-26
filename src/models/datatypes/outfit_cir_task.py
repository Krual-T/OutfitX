from typing import List
from pydantic import BaseModel, Field
from .fashion_item import FashionItem


class OutfitComplementaryItemRetrievalTask(BaseModel):
    outfit: List[FashionItem] = Field(
        default_factory=list,
        description="List of fashion items"
    )
    target_item: FashionItem = Field(
        default_factory=FashionItem,
        description="With embedding of description (no image) of target item."
    )
