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
        batch_size = len(batch)
        queries_iter, labels_iter = zip(*batch)
        outfit_seq = [query.outfit for query in queries_iter]
        max_length = self._get_max_length(sequences=outfit_seq)
        image_sequences = self._pad_sequences(
            sequences=[
                [
                    self.transform(item.image if not isinstance(item.image, np.ndarray) else Image.fromarray(item.image) )
                    for item in outfit
                ] for outfit in outfit_seq
            ],
            pad_value = self.transform(self.image_pad),
            max_length=max_length,
            return_tensor=True
        )
        text_sequences = self._pad_sequences(
            sequences=[
                [item.category for item in outfit]
                for outfit in outfit_seq
            ],
            max_length=max_length,
            pad_value = self.text_pad
        )
        texts = flatten_seq_to_one_dim(text_sequences)
        inputs = self.tokenizer(
            texts, **self.tokenizer_kargs
        )
        text_sequences = {
            k: v.view(batch_size * max_length, *v.size()[2:]) for k, v in inputs.items()
        }
        item_length = lambda seq: min(len(seq), max_length)
        pad_length = lambda seq: max_length - item_length(seq)
        mask = torch.tensor(
            data=[
                [0] * item_length(sequence) + [1] * (pad_length(sequence))
                for sequence in outfit_seq
            ],
            dtype=torch.bool
        )
        encoder_input_dict = {
                'images': image_sequences,
                'texts': text_sequences,
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