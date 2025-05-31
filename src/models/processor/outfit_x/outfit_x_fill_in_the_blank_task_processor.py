import torch

from src.models.datatypes import OutfitComplementaryItemRetrievalTask
from .outfit_x_base_processor import OutfitXBaseProcessor


class OutfitXFillInTheBlankTaskProcessor(OutfitXBaseProcessor):

    def __call__(self, batch):
        queries_iter, candidate_item_embeddings_iter, batch_y_iter = zip(*batch)

        queries = [query for query in queries_iter]
        input_dict = self._build_input_dict(queries)
        batch_dict = {
            'input_dict': input_dict,
            'candidate_item_embedding': torch.stack(candidate_item_embeddings_iter),
            'answer_index': torch.tensor(batch_y_iter, dtype=torch.long)
        }# candidates [batch_size, 4, 768]
        return batch_dict

    def _build_input_dict(self, queries):
        # input_dict
        outfit_sequence = [[item.embedding for item in query.outfit] for query in queries]
        outfit_embedding_batch, outfits_mask = self._to_tensor_and_padding(
            sequences=outfit_sequence
        )

        target_item_text_embedding_batch = torch.stack([
            torch.tensor(
                data=query.target_item.text_embedding,
                dtype=torch.float
            ) for query in queries
        ]) # (B, d_embed)

        return {
            'task': OutfitComplementaryItemRetrievalTask,
            'outfit_embedding': outfit_embedding_batch,
            'outfit_mask': outfits_mask,
            'target_item_text_embedding':target_item_text_embedding_batch
        }