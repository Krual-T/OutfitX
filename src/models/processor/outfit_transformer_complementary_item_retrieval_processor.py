import torch
from typing import Literal
from .outfit_transformer_base_processor import OutfitTransformerBaseProcessor
from src.models.datatypes import OutfitCompatibilityPredictionTask, OutfitComplementaryItemRetrievalTask


class OutfitTransformerComplementaryItemRetrievalTaskProcessor(OutfitTransformerBaseProcessor):
    def __init__(self, run_mode:Literal['train', 'valid', 'test'],*args, **kwargs):
        super().__init__(*args, **kwargs)
        if run_mode == 'train':
            self.__call_mode = self.__train_call
        elif run_mode == 'valid':
            self.__call_mode = self.__valid_call
        elif run_mode == 'test':
            self.__call_mode = self.__test_call


    def __call__(self, batch):
        return self.__call_mode(batch)

    def __train_call(self, batch):
        query_iter, neg_items_emb_iter = zip(*batch)
        queries = [query for query in query_iter]
        sequences = [query.outfit for query in queries]
        target_time_text_embedding_batch = [query.target_item.text_embedding for query in queries]
        outfits_embedding, outfits_mask = self._get_embeddings_and_padding_masks(
            sequences=sequences
        )
        pos_item_embedding_batch = torch.stack([
            torch.tensor(
                query.target_item.embedding,
                dtype=torch.float,
            )
            for query in queries
        ])
        neg_items_embedding_batch = torch.stack([
            torch.stack([
                torch.tensor(
                    item_emb,
                    dtype=torch.float,
                )
                for item_emb in neg_items_emb
            ])
            for neg_items_emb in neg_items_emb_iter
        ])
        input_dict = {
            'task': OutfitCompatibilityPredictionTask,
            'outfit': outfits_embedding,
            'outfit_mask': outfits_mask,
            'target_item_text_embedding':target_time_text_embedding_batch
        }
        batch_dict = {
            'input_dict': input_dict,
            'pos_item_embedding': pos_item_embedding_batch,
            'neg_items_embedding': neg_items_embedding_batch,
        }
        return batch_dict

    def __valid_call(self, batch):
        query_iter, neg_items_emb_iter = zip(*batch)
        queries = [query for query in query_iter]
        target_time_text_embedding_batch = [query.target_item.text_embedding for query in queries]

        pos_item_id_batch = [query.target_item.item_id for query in queries]
        pos_item_embedding_batch = torch.stack([
                torch.tensor(
                    query.target_item.embedding,
                    dtype=torch.float
                )
                for query in queries
            ])
        neg_items_embedding_batch = torch.stack([
            torch.stack([
                torch.tensor(
                    item_emb,
                    dtype=torch.float
                )
                for item_emb in neg_items_emb
            ])
            for neg_items_emb in neg_items_emb_iter
        ])
        sequences = [query.outfit for query in queries]
        outfits_embedding, outfits_mask = self._get_embeddings_and_padding_masks(
            sequences=sequences
        )
        input_dict = {
            'task': OutfitComplementaryItemRetrievalTask,
            'outfit': outfits_embedding,
            'outfit_mask': outfits_mask,
            'target_item_text_embedding': target_time_text_embedding_batch,
        }
        batch_dict = {
            'input_dict': input_dict,
            'pos_item_id': pos_item_id_batch,
            'pos_item_embedding': pos_item_embedding_batch,
            'neg_items_embedding': neg_items_embedding_batch,
        }
        return batch_dict

    def __test_call(self, batch):
        query_iter, _ = zip(*batch)
        queries = [query for query in query_iter]
        target_time_text_embedding_batch = [query.target_item.text_embedding for query in queries]
        pos_item_id_batch = [query.target_item.item_id for query in queries]
        sequences = [query.outfit for query in queries]
        outfits_embedding_batch, outfits_mask_batch = self._get_embeddings_and_padding_masks(
            sequences=sequences
        )
        input_dict = {
            'task': OutfitComplementaryItemRetrievalTask,
            'outfit': outfits_embedding_batch,
            'outfit_mask': outfits_mask_batch,
            'target_item_text_embedding': target_time_text_embedding_batch,
        }
        batch_dict = {
            'input_dict': input_dict,
            'pos_item_id': pos_item_id_batch,
        }
        return batch_dict