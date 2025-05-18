from pydantic import BaseModel
from src.models.datatypes import FashionItem


class OutfitPrecomputeEmbeddingTask(BaseModel):
    fashion_item: FashionItem