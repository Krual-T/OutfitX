import torch

from .outfit_transformer_base_processor import OutfitTransformerBaseProcessor
from src.models.datatypes import OutfitCompatibilityPredictionTask

class OutfitTransformerOriginalCompatibilityPredictionTaskProcessor(OutfitTransformerBaseProcessor):
    def __call__(self, batch):
        queries_iter, labels_iter = zip(*batch)
        outfit_seq = [query.outfit for query in queries_iter]
        max_length = self._get_max_length(sequences=outfit_seq)
        image_sequences = self._pad_sequences(
            sequences=[[item.image for item in outfit] for outfit in outfit_seq],
            pad_value = self.image_pad,
            max_length=max_length
        )
        text_sequences = self._pad_sequences(
            sequences=[[item.category for item in outfit] for outfit in outfit_seq],
            max_length=max_length,
            pad_value = self.text_pad
        )
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