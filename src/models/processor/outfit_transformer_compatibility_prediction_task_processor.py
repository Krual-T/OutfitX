import torch

from .outfit_transformer_base_processor import OutfitTransformerBaseProcessor
from src.models.datatypes import OutfitCompatibilityPredictionTask

class OutfitTransformerCompatibilityPredictionTaskProcessor(OutfitTransformerBaseProcessor):
    def __call__(self, batch):
        queries_iter, labels_iter = zip(*batch)
        sequences = [query.outfit for query in queries_iter]
        outfit_embedding_batch,outfits_mask_batch = self._get_embeddings_and_padding_masks(
            sequences=sequences
        )
        input_dict = {
                'task': OutfitCompatibilityPredictionTask,
                'outfit_embedding': outfit_embedding_batch,
                'outfit_mask': outfits_mask_batch,
            }
        batch_dict = {
            'input_dict': input_dict,
            'label': torch.tensor(labels_iter, dtype=torch.float)
        }
        return batch_dict