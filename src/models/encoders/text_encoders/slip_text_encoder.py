from typing import List

import open_clip
import torch
from torch import autocast

from src.models.encoders.base_encoders import BaseTextEncoder
from src.models.utils.model_utils import freeze_model


class SigLIPTextEncoder(BaseTextEncoder):
    def __init__(
            self,
            model_name_or_path: str = "hf-hub:Marqo/marqo-fashionSigLIP",
            freeze: bool = True
    ):
        super().__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(model_name_or_path)
        self.model.eval()
        if freeze:
            freeze_model(self.model)
        self.tokenizer = open_clip.get_tokenizer(model_name_or_path)
    @torch.no_grad()
    def _forward(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(texts).to(self.device)
        with autocast(device_type=self.device.type, dtype=torch.float16):
            text_embeddings = self.model.encode_text(inputs)
        return text_embeddings

    @property
    def d_embed(self) -> int:
        if self.model_name_or_path == "hf-hub:Marqo/marqo-fashionSigLIP":
            return 768
        else:
            raise NotImplementedError