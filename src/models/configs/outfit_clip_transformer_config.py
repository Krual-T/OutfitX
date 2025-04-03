from dataclasses import dataclass
from outfit_transformer_config import OutfitTransformerConfig

@dataclass
class OutfitCLIPTransformerConfig(OutfitTransformerConfig):
    item_enc_clip_model_name: str = "patrickjohncyh/fashion-clip"