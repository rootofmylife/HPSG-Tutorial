import torch
import torch.nn as nn
from torch import from_numpy

import numpy as np

from model.multilevel_embedding import MultiLevelEmbedding
from model.encoder import Encoder
from model.layer_normalization import LayerNormalization
from model.dep_score import DepScore

class ChartParser(nn.Module):
    def __init__(self, tag_vocab, word_vocab, label_vocab, char_vocab, type_vocab):
        super().__init__()

        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.char_vocab = char_vocab
        self.type_vocab = type_vocab

        self.d_model = 1024
        self.partitioned = True
        self.d_content = (self.d_model // 2) if self.partitioned else self.d_model
        self.d_positional = (self.d_model // 2) if self.partitioned else None

        # Use LAL
        self.use_lal = True
        self.lal_d_kv = 64
        self.lal_d_proj = 64
        self.lal_resdrop = True
        self.lal_pwff = True
        self.lal_q_as_matrix = False
        self.lal_partitioned = True
        self.lal_combine_as_self = False

        num_embeddings_map = {
            'tags': self.tag_vocab.size,
            'words': self.word_vocab.size,
            'chars': self.char_vocab.size,
        }

        emb_dropouts_map = {
            'tags': 0.2,
            'words': 0.4,
        }

        self.emb_types = ['tags', 'words']

        self.embedding = MultiLevelEmbedding(
            [num_embeddings_map[emb_type] for emb_type in self.emb_types],
            self.d_model,
            d_positional=self.d_positional,
            dropout=0.2,
            timing_dropout=0.0,
            emb_dropouts_list=[emb_dropouts_map[emb_type] for emb_type in self.emb_types],
            extra_content_dropout=None,
            max_len=300,
            word_table_np=None,
        )

        self.encoder = Encoder(
            self.embedding,
            num_layers=12,
            num_heads=8,
            d_kv=64,
            d_ff=2048,
            d_positional=self.d_positional,
            relu_dropout=0.2,
            residual_dropout=0.2,
            attention_dropout=0.2,
            use_lal=True,
            lal_d_kv=64,
            lal_d_proj=64,
            lal_resdrop=True,
            lal_pwff=True,
            lal_q_as_matrix=False,
            lal_partitioned=True,
        )

        annotation_dim = ((label_vocab.size - 2) * self.lal_d_proj) if (self.use_lal and not self.lal_combine_as_self) else self.d_model

        self.f_label = nn.Sequential(
            nn.Linear(annotation_dim, 250),
            LayerNormalization(250),
            nn.ReLU(),
            nn.Linear(250, label_vocab.size - 1 ),
        )
        self.dep_score = DepScore(type_vocab.size, annotation_dim)
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
