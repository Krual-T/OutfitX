from torch import nn
from .image_encoders import Resnet18ImageEncoder,CLIPImageEncoder
from .text_encoders import CLIPTextEncoder,HuggingFaceTextEncoder

from src.models.utils.model_utils import aggregate_embeddings
from src.models.configs import ItemEncoderConfig
class ItemEncoder(nn.Module):
    def __init__(self,cfg: ItemEncoderConfig):
        super().__init__()
        self.cfg = cfg
        if self.cfg.type == 'resnet_hf_sentence_bert':
            self.image_enc = Resnet18ImageEncoder(
                d_embed=self.cfg.dim_per_modality,
            )
            self.text_enc = HuggingFaceTextEncoder(
                d_embed=self.cfg.dim_per_modality,
                model_name_or_path=cfg.text_model_name
            )
        elif cfg.type == 'clip':
            self.image_enc = CLIPImageEncoder(
                model_name_or_path=cfg.clip_model_name
            )
            self.text_enc = CLIPTextEncoder(
                model_name_or_path=cfg.clip_model_name
            )

    @property
    def d_embed(self):
        return self.cfg.dim_per_modality * 2 if self.cfg.aggregation_method == 'concat' else self.cfg.dim_per_modality

    @property
    def image_size(self):
        return self.image_enc.image_size

    def forward(self, images, texts, *args, **kwargs):
        # Encode images and texts
        image_embeddings = self.image_enc(
            images, normalize=self.cfg.norm_out, *args, **kwargs
        )
        text_embeddings = self.text_enc(
            texts, normalize=self.cfg.norm_out, *args, **kwargs
        )
        # Aggregate embeddings
        encoder_outputs = aggregate_embeddings(
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            aggregation_method=self.cfg.aggregation_method
        )

        return encoder_outputs