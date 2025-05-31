import torch
from typing import Literal
from .outfit_x_base_processor import OutfitXBaseProcessor
from src.models.datatypes import OutfitComplementaryItemRetrievalTask


class OutfitXComplementaryItemRetrievalTaskProcessor(OutfitXBaseProcessor):
    def __init__(self, run_mode:Literal['train', 'valid', 'test'],*args, **kwargs):
        super().__init__(*args, **kwargs)
        if run_mode == 'train':
            self.__call_mode = self._train_call
        elif run_mode == 'valid':
            self.__call_mode = self._valid_call
        elif run_mode == 'test':
            self.__call_mode = self._test_call

    def __call__(self, batch):
        return self.__call_mode(batch)

    def _train_call(self, batch):
        query_iter, neg_items_emb_iter = zip(*batch)
        queries = [query for query in query_iter]
        # input_dict
        input_dict = self._build_input_dict(queries)
        # metric parameters
        pos_item_embedding_batch = torch.stack([
            torch.tensor(
                query.target_item.embedding,
                dtype=torch.float,
            )
            for query in queries
        ])
        neg_items_embedding_batch, neg_items_mask_batch = self._to_tensor_and_padding(
            sequences=[
                [item_emb for item_emb in neg_items_emb]
                for neg_items_emb in neg_items_emb_iter
            ]
        )
        # neg_items_embedding_batch = torch.stack([
        #     torch.stack([
        #         torch.tensor(
        #             item_emb,
        #             dtype=torch.float,
        #         )
        #         for item_emb in neg_items_emb
        #     ])
        #     for neg_items_emb in neg_items_emb_iter
        # ])
        batch_dict = {
            'input_dict': input_dict,
            'pos_item_embedding': pos_item_embedding_batch,
            'neg_items_embedding': neg_items_embedding_batch,
            'neg_items_mask': neg_items_mask_batch,
        }
        return batch_dict

    def _valid_call(self, batch):
        query_iter, neg_items_emb_iter = zip(*batch)
        queries = [query for query in query_iter]
        input_dict = self._build_input_dict(queries)
        pos_item_id_batch = [query.target_item.item_id for query in queries]
        pos_item_embedding_batch = torch.stack([
                torch.tensor(
                    query.target_item.embedding,
                    dtype=torch.float
                )
                for query in queries
            ])
        neg_items_embedding_batch, neg_items_mask_batch = self._to_tensor_and_padding(
            sequences=[
                [item_emb for item_emb in neg_items_emb]
                for neg_items_emb in neg_items_emb_iter
            ]
        )
        batch_dict = {
            'input_dict': input_dict,
            'pos_item_id': pos_item_id_batch,
            'pos_item_embedding': pos_item_embedding_batch,
            'neg_items_embedding': neg_items_embedding_batch,
            'neg_items_mask': neg_items_mask_batch,
        }
        return batch_dict

    def _test_call(self, batch):
        query_iter, _ = zip(*batch)
        queries = [query for query in query_iter]
        input_dict = self._build_input_dict(queries)
        pos_item_id_batch = [query.target_item.item_id for query in queries]
        batch_dict = {
            'input_dict': input_dict,
            'pos_item_id': pos_item_id_batch,
        }
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