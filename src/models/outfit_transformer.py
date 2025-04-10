from torch import nn
from configs import OutfitTransformerConfig
from encoders import ItemEncoder
class OutfitTransformer(nn.Module):
    def __init__(self, cfg: OutfitTransformerConfig= None):
        super().__init__()
        self.cfg = cfg if cfg is not None else OutfitTransformerConfig()
        self._init_item_enc()
        self._init_style_enc()
        self._init_variables()

    def _init_item_enc(self):
        """Builds the outfit encoders using configuration parameters."""
        self.item_enc = ItemEncoder(
            text_model_name=self.cfg.item_enc_text_model_name,
            enc_dim_per_modality=self.cfg.item_enc_dim_per_modality,
            enc_norm_out=self.cfg.item_enc_norm_out,
            aggregation_method=self.cfg.aggregation_method
        )