from torch import nn
from image_encoders import Resnet18ImageEncoder
from text_encoders import HuggingFaceTextEncoder
from src.models.utils.model_utils import aggregate_embeddings

class ItemEncoder(nn.Module):
    def __init__(
            self,
            model_name,
            enc_dim_per_modality,
            enc_norm_out,
            aggregation_method
    ):
        super().__init__()
        self.enc_dim_per_modality = enc_dim_per_modality
        self.aggregation_method = aggregation_method
        self.enc_norm_out = enc_norm_out
        self._build_encoders(model_name)

    def _build_encoders(self, model_name):
        self.image_enc = Resnet18ImageEncoder(
            embedding_size=self.enc_dim_per_modality,
        )
        self.text_enc = HuggingFaceTextEncoder(
            embedding_size=self.enc_dim_per_modality,
            model_name_or_path=model_name
        )

    @property
    def d_embed(self):
        if self.aggregation_method == 'concat':
            d_model = self.enc_dim_per_modality * 2
        else:
            d_model = self.enc_dim_per_modality

        return d_model

    @property
    def image_size(self):
        return self.image_enc.image_size

    def forward(self, images, texts, *args, **kwargs):
        # Encode images and texts
        image_embeddings = self.image_enc(
            images, normalize=self.enc_norm_out, *args, **kwargs
        )
        text_embeddings = self.text_enc(
            texts, normalize=self.enc_norm_out, *args, **kwargs
        )
        # Aggregate embeddings
        encoder_outputs = aggregate_embeddings(
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            aggregation_method=self.aggregation_method
        )

        return encoder_outputs