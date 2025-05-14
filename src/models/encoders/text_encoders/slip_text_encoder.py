from typing import List

import open_clip
import torch

from src.models.encoders.base_encoders import BaseTextEncoder
from src.models.utils.model_utils import freeze_model


class SigLIPTextEncoder(BaseTextEncoder):
    def __init__(
            self,
            model_name_or_path: str = 'Marqo/marqo-fashionSigLIP',
            freeze: bool = True
    ):
        super().__init__()
        self.max_length = 64
        self.model, _, _ = open_clip.create_model_and_transforms(model_name_or_path)
        self.model.eval()
        if freeze:
            freeze_model(self.model)
        self.tokenizer = open_clip.get_tokenizer(model_name_or_path)
    def _forward(self, texts: List[str]) -> torch.Tensor:
        tokenizer_inputs = {
            'text': texts,
            'max_length': self.max_length,
            'padding': 'max_length',
            'truncation': True,
            'return_tensors': 'pt'
        }
        text = self.tokenizer(**tokenizer_inputs)
        text = {
            k: v.to(self.device)
            for k, v in text.items()
        }
        text_embeddings = self.model.encode_text(**text)
        return text_embeddings

    @property
    def d_embed(self) -> int:
        if self.model_name_or_path == "Marqo/marqo-fashionSigLIP":
            return 768
        else:
            raise NotImplementedError