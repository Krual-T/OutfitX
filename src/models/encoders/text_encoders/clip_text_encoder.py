import torch

from torch import Tensor
from transformers import CLIPTokenizer, CLIPTextModelWithProjection
from typing import List
from typing import Dict, Any
from src.models.encoders.base_encoders import BaseTextEncoder
from src.utils.model_utils import freeze_model


class CLIPTextEncoder(BaseTextEncoder):

    def __init__(
            self,
            model_name_or_path: str = 'patrickjohncyh/fashion-clip',
            freeze: bool = True
    ):
        super().__init__()
        self.model = CLIPTextModelWithProjection.from_pretrained(
            model_name_or_path, weights_only=False
        )
        self.model.eval()
        if freeze:
            freeze_model(self.model)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name_or_path
        )

    # 输出维度512
    @property
    def d_embed(self) -> int:
        return self.model.config.projection_dim

    @torch.no_grad()
    def _forward(
            self,
            texts: List[str],
            tokenizer_kargs: Dict[str, Any] = None
    ) -> Tensor:

        tokenizer_kargs = tokenizer_kargs if tokenizer_kargs is not None else {
            'max_length': 64,
            'padding': 'max_length',
            'truncation': True,
        }
        tokenizer_kargs['return_tensors'] = 'pt'
        inputs = self.tokenizer(
            text=texts, **tokenizer_kargs
        )
        # {input_ids: torch.Tensor, attention_mask: torch.Tensor}
        inputs = {
            key: value.to(self.device) for key, value in inputs.items()
        }

        text_embeddings = self.model(
            **inputs
        ).text_embeds

        return text_embeddings