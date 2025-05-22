import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms
from transformers import AutoTokenizer

from src.utils.model_utils import flatten_seq_to_one_dim
from .outfit_transformer_base_processor import OutfitTransformerBaseProcessor
from src.models.datatypes import OutfitCompatibilityPredictionTask

class OutfitTransformerOriginalCompatibilityPredictionTaskProcessor(OutfitTransformerBaseProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.tokenizer_kargs = {
            'max_length': 32,
            'padding': 'max_length',
            'truncation': True,
            'return_tensors':'pt'
        }
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.item_encoder.text_model_name)
    def __call__(self, batch):
        item_length = lambda seq: min(len(seq), max_length)
        pad_length = lambda seq: max_length - item_length(seq)

        batch_size = len(batch)
        queries_iter, labels_iter = zip(*batch)
        outfit_seq = [query.outfit for query in queries_iter]
        max_length = self._get_max_length(sequences=outfit_seq)

        img_pad_value = self.transform(self.image_pad)
        image_seq = self._pad_sequences(
            sequences=[[item.image for item in outfit] for outfit in outfit_seq],
            max_length=max_length,
            pad_value=img_pad_value,
            return_tensor=False
        )
        images_tensor = torch.stack([
            torch.stack([item_img for item_img in outfit]) # item_img是tensor
            for outfit in image_seq
        ])
        text_seq = self._pad_sequences(
            sequences =[[item.category for item in outfit]for outfit in outfit_seq],
            max_length=max_length,
            pad_value=self.text_pad,
            return_tensor=False
        )
        texts = flatten_seq_to_one_dim([
            [item.category for item in outfit]
            for outfit in outfit_seq
        ])# (B,max_L) ->（B*max_length）
        inputs = self.tokenizer(
            texts, **self.tokenizer_kargs
        )# （B*max_length）->（B*max_length,token_length）
        texts_tensor = {
            k: v.view(batch_size,max_length, *v.size()[2:]) for k, v in inputs.items()
        }

        mask = torch.tensor(
            data=[
                [0] * item_length(sequence) + [1] * (pad_length(sequence))
                for sequence in outfit_seq
            ],
            dtype=torch.bool
        )
        encoder_input_dict = {
                'images': images_tensor,
                'texts': texts_tensor,
            }
        cp_input_dict = {
                'task': OutfitCompatibilityPredictionTask,
                'outfit_embedding': None,
                'outfit_mask': mask,
            }
        batch_dict = {
            'cp_input_dict': cp_input_dict,
            'encoder_input_dict': encoder_input_dict,
            'label': torch.tensor(labels_iter, dtype=torch.float)
        }
        return batch_dict