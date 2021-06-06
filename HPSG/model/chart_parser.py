import torch
import torch.nn as nn
from torch import from_numpy

import numpy as np

from multilevel_embedding import MultiLevelEmbedding
from encoder import Encoder
from layer_normalization import LayerNormalization
from dep_score import DepScore

class ChartParser(nn.Module):
    def __init__(self, tag_vocab, word_vocab, label_vocab, char_vocab, type_vocab, params) -> None:
        super().__init__()

        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.char_vocab = char_vocab
        self.type_vocab = type_vocab

        self.params = params

        self.d_model = params.d_model
        self.partitioned = params.partitioned
        self.d_content = (self.d_model // 2) if self.partitioned else self.d_model
        self.d_positional = (params.d_model // 2) if self.partitioned else None

        # Use LAL
        self.lal_d_kv = params.lal_d_kv
        self.lal_d_proj = params.lal_d_proj
        self.lal_resdrop = params.lal_resdrop
        self.lal_pwff = params.lal_pwff
        self.lal_q_as_matrix = params.lal_q_as_matrix
        self.lal_partitioned = params.lal_partitioned
        self.lal_combine_as_self = params.lal_combine_as_self

        num_embeddings_map = {
            'tags': tag_vocab.size,
            'words': word_vocab.size,
            'chars': char_vocab.size,
        }

        emb_dropouts_map = {
            'tags': params.tag_emb_dropout,
            'words': params.word_emb_dropout,
        }

        self.emb_types = ['tags', 'words']

        self.embedding = MultiLevelEmbedding(
            [num_embeddings_map[emb_type] for emb_type in self.emb_types],
            params.d_model,
            params=params,
            d_positional=self.d_positional,
            dropout=params.embedding_dropout,
            timing_dropout=params.timing_dropout,
            emb_dropouts_list=[emb_dropouts_map[emb_type] for emb_type in self.emb_types],
            extra_content_dropout=self.morpho_emb_dropout,
            max_len=params.sentence_max_len,
            word_table_np=None,
        )

        self.encoder = Encoder(
            params,
            self.embedding,
            num_layers=params.num_layers,
            num_heads=params.num_heads,
            d_kv=params.d_kv,
            d_ff=params.d_ff,
            d_positional=self.d_positional,
            relu_dropout=params.relu_dropout,
            residual_dropout=params.residual_dropout,
            attention_dropout=params.attention_dropout,
            use_lal=params.use_lal,
            lal_d_kv=params.lal_d_kv,
            lal_d_proj=params.lal_d_proj,
            lal_resdrop=params.lal_resdrop,
            lal_pwff=params.lal_pwff,
            lal_q_as_matrix=params.lal_q_as_matrix,
            lal_partitioned=params.lal_partitioned,
        )

        annotation_dim = ((label_vocab.size - 2) * self.lal_d_proj) if (self.use_lal and not self.lal_combine_as_self) else params.d_model
        params.annotation_dim = annotation_dim

        self.f_label = nn.Sequential(
            nn.Linear(annotation_dim, params.d_label_hidden),
            LayerNormalization(params.d_label_hidden),
            nn.ReLU(),
            nn.Linear(params.d_label_hidden, label_vocab.size - 1 ),
        )
        self.dep_score = DepScore(params, type_vocab.size)
        self.loss_func = torch.nn.CrossEntropyLoss(size_average=False)
        self.loss_funt = torch.nn.CrossEntropyLoss(size_average=False)

class BatchIndices:
    """
    Batch indices container class (used to implement packed batches)
    """
    def __init__(self, batch_idxs_np):
        self.batch_idxs_np = batch_idxs_np
        self.batch_idxs_torch = from_numpy(batch_idxs_np)

        self.batch_size = int(1 + np.max(batch_idxs_np))

        batch_idxs_np_extra = np.concatenate([[-1], batch_idxs_np, [-1]])
        self.boundaries_np = np.nonzero(batch_idxs_np_extra[1:] != batch_idxs_np_extra[:-1])[0]
        self.seq_lens_np = self.boundaries_np[1:] - self.boundaries_np[:-1]
        assert len(self.seq_lens_np) == self.batch_size
        self.max_len = int(np.max(self.boundaries_np[1:] - self.boundaries_np[:-1]))
