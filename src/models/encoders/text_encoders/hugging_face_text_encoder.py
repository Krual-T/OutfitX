import torch
import torch.nn as nn

from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from typing import List, Union
from typing import Dict, Any

from src.models.encoders.base_encoders import BaseTextEncoder
from src.utils.model_utils import freeze_model, mean_pooling


class HuggingFaceTextEncoder(BaseTextEncoder):
    # default model：SentenceBERT——paper's implement
    def __init__(
            self,
            d_embed: int = 64,
            model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
            freeze: bool = True
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path)
        if freeze:
            freeze_model(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # projection
        self.proj = nn.Linear(
            in_features=self.model.config.hidden_size,
            out_features=d_embed
        )

    @property
    def d_embed(self) -> int:
        return self.proj.out_features

    @torch.no_grad()
    def _forward(
            self,
            texts: Union[List[str], dict],
            tokenizer_kargs: Dict[str, Any] = None
    ) -> Tensor:

        if isinstance(texts[0], str):
            tokenizer_kargs = tokenizer_kargs if tokenizer_kargs is not None else {
                'max_length': 32,
                'padding': 'max_length',
                'truncation': True,
            }

            tokenizer_kargs['return_tensors'] = 'pt'
            inputs = self.tokenizer(
                texts, **tokenizer_kargs
            )
        elif isinstance(texts, dict) and 'input_ids' in texts and 'attention_mask' in texts:
            inputs = texts
        else:
            raise ValueError("Invalid input format")
        inputs = {
            key: value.to(self.device) for key, value in inputs.items()
        }
        outputs = mean_pooling(
            model_output=self.model(**inputs),
            attention_mask=inputs['attention_mask']
        )
        text_embeddings = self.proj(
            outputs
        )


        return text_embeddings