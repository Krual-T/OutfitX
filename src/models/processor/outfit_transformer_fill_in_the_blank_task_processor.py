import torch

from src.models.datatypes import OutfitComplementaryItemRetrievalTask
from .outfit_transformer_base_processor import OutfitTransformerBaseProcessor


class OutfitTransformerFillInTheBlankTaskProcessor(OutfitTransformerBaseProcessor):

    def __call__(self, batch):
        queries_iter, candidate_item_embeddings_iter, batch_y_iter = zip(*batch)
        sequences = [query.outfit for query in queries_iter]
        outfit_embedding_batch, outfit_mask_batch = self._get_embeddings_and_padding_masks(
            sequences=sequences
        )
        input_dict = {
            'type': OutfitComplementaryItemRetrievalTask,
            'outfit': outfit_embedding_batch,
            'outfit_mask': outfit_mask_batch
        }
        batch_dict = {
            'input_dict': input_dict,
            'candidate_item_embedding': torch.stack(candidate_item_embeddings_iter),
            'answer_index': torch.tensor(batch_y_iter, dtype=torch.long)
        }# candidates [batch_size, 4, 768]
        return batch_dict
