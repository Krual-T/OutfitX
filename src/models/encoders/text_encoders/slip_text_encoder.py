from typing import List, Dict, Any

import open_clip
import torch
from torch import autocast

from src.models.encoders.base_encoders import BaseTextEncoder
from src.utils.model_utils import freeze_model


class SigLIPTextEncoder(BaseTextEncoder):
    def __init__(
            self,
            model_context: Dict[str, Any],
    ):
        super().__init__()
        self.model_name_or_path = model_context.get('model_name_or_path',"hf-hub:Marqo/marqo-fashionSigLIP")
        if 'model' not in model_context:
            self.model = open_clip.create_model(self.model_name_or_path)
            self.model.eval()
            freeze_model(self.model)
        else:
            self.model = model_context.get('model')
        self.tokenizer = model_context.get('tokenizer', open_clip.get_tokenizer(self.model_name_or_path))


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