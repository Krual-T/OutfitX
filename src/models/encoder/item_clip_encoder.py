from item_encoder import ItemEncoder
from image_encoder import CLIPImageEncoder
from text_encoder import CLIPTextEncoder
class CLIPItemEncoder(ItemEncoder):
    def __init__(
            self,
            model_name,
            enc_norm_out,
            aggregation_method
    ):
        super().__init__(
            model_name=model_name,
            enc_dim_per_modality=512,
            enc_norm_out=enc_norm_out,
            aggregation_method=aggregation_method
        )

    def _build_encoders(self, model_name):
        self.image_enc = CLIPImageEncoder(
            model_name_or_path=model_name
        )
        self.text_enc = CLIPTextEncoder(
            model_name_or_path=model_name
        )