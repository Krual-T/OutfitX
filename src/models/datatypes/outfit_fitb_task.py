from typing import List
from pydantic import BaseModel, Field
from .fashion_item import FashionItem
# 这个任务类和CIR任务类（OutfitComplementaryItemRetrievalTask）一样
# 只是区分任务类型
class OutfitFillInTheBlankTask(BaseModel):
    outfit: List[FashionItem] = Field(
        default_factory=list,
        description="List of fashion items"
    )
    # 在Outfit内部只用了target_item的description的embedding（text_embedding），image的被剥离了
    target_item: FashionItem = Field(
        default_factory=FashionItem,
        description="With embedding of description (no image) of target item."
    )
    def __len__(self):
        return len(self.outfit)