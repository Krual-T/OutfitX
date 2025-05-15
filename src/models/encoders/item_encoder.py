import open_clip
from torch import nn
from .image_encoders import Resnet18ImageEncoder, CLIPImageEncoder, SigLIPImageEncoder
from .text_encoders import CLIPTextEncoder, HuggingFaceTextEncoder, SigLIPTextEncoder

from src.models.utils.model_utils import aggregate_embeddings, freeze_model
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
        elif cfg.type == 'slip':
            slip_model, _, preprocess_val = open_clip.create_model_and_transforms(cfg.slip_model_name)
            model_context = {
                'model':slip_model,
                'preprocess_val':preprocess_val,
                'model_name_or_path': cfg.slip_model_name
            }
            model_context['model'].eval()
            freeze_model(model_context['model'])
            self.image_enc = SigLIPImageEncoder(model_context=model_context)
            self.text_enc = SigLIPTextEncoder(model_context=model_context)
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