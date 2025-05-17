from typing import List, Union
from pydantic import BaseModel, Field
from .fashion_item import FashionItem

class OutfitCompatibilityPredictionTask(BaseModel):
    outfit: List[FashionItem] = Field(
        default_factory=list,
        description="List of fashion items"
    )
    def __len__(self):
        return len(self.outfit)