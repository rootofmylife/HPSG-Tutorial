import torch
import torch.nn as nn
import torch.nn.init as init

from feaure_dropout import FeatureDropout
from layer_normalization import LayerNormalization

class MultiLevelEmbedding(nn.Module):
    def __init__(self,
            num_embeddings_list,
            d_embedding,
            hparams,
            d_positional=None,
            max_len=300,
            normalize=True,
            dropout=0.1,
            timing_dropout=0.0,
            emb_dropouts_list=None,
            extra_content_dropout=None,
            word_table_np = None,
            **kwargs):
        super().__init__()

        self.d_embedding = d_embedding
        self.partitioned = d_positional is not None
        self.hparams = hparams

        if self.partitioned:
            self.d_positional = d_positional
            self.d_content = self.d_embedding - self.d_positional
        else:
            self.d_positional = self.d_embedding
            self.d_content = self.d_embedding

        if emb_dropouts_list is None:
            emb_dropouts_list = [0.0] * len(num_embeddings_list)
        assert len(emb_dropouts_list) == len(num_embeddings_list)

        if word_table_np is not None:
            self.pretrain_dim = word_table_np.shape[1]
        else:
            self.pretrain_dim = 0

        embs = []
        emb_dropouts = []
        cun = len(num_embeddings_list)*2
        for i, (num_embeddings, emb_dropout) in enumerate(zip(num_embeddings_list, emb_dropouts_list)):
            if hparams.use_cat:
                if i == len(num_embeddings_list) - 1:
                    #last is word
                    emb = nn.Embedding(num_embeddings, self.d_content//cun - self.pretrain_dim, **kwargs)
                else :
                    emb = nn.Embedding(num_embeddings, self.d_content//cun, **kwargs)
            else :
                emb = nn.Embedding(num_embeddings, self.d_content - self.pretrain_dim, **kwargs)
            embs.append(emb)
            emb_dropout = FeatureDropout(emb_dropout)
            emb_dropouts.append(emb_dropout)

        if word_table_np is not None:
            self.pretrain_emb = nn.Embedding(word_table_np.shape[0], self.pretrain_dim)
            self.pretrain_emb.weight.data.copy_(torch.from_numpy(word_table_np))
            self.pretrain_emb.weight.requires_grad_(False)
            self.pretrain_emb_dropout = FeatureDropout(0.33)

        self.embs = nn.ModuleList(embs)
        self.emb_dropouts = nn.ModuleList(emb_dropouts)

        if extra_content_dropout is not None:
            self.extra_content_dropout = FeatureDropout(extra_content_dropout)
        else:
            self.extra_content_dropout = None

        if normalize:
            self.layer_norm = LayerNormalization(d_embedding)
        else:
            self.layer_norm = lambda x: x

        self.dropout = FeatureDropout(dropout)
        self.timing_dropout = FeatureDropout(timing_dropout)

        # Learned embeddings
        self.max_len = max_len
        self.position_table = nn.Parameter(torch.FloatTensor(max_len, self.d_positional))
        init.normal_(self.position_table)

    def forward(self, xs, pre_words_idxs, batch_idxs, extra_content_annotations=None):
        content_annotations = [
            emb_dropout(emb(x), batch_idxs)
            for x, emb, emb_dropout in zip(xs, self.embs, self.emb_dropouts)
            ]
        if self.hparams.use_cat:
            content_annotations = torch.cat(content_annotations, dim = -1)
        else :
            content_annotations = sum(content_annotations)
        if self.pretrain_dim != 0:
            content_annotations = torch.cat([content_annotations, self.pretrain_emb_dropout(self.pretrain_emb(pre_words_idxs), batch_idxs)], dim  = 1)

        if extra_content_annotations is not None:
            if self.extra_content_dropout is not None:
                extra_content_annotations = self.extra_content_dropout(extra_content_annotations, batch_idxs)

            if self.hparams.use_cat:
                content_annotations = torch.cat(
                    [content_annotations, extra_content_annotations], dim=-1)
            else:
                content_annotations += extra_content_annotations

        timing_signal = []
        for seq_len in batch_idxs.seq_lens_np:
            this_seq_len = seq_len
            timing_signal.append(self.position_table[:this_seq_len,:])
            this_seq_len -= self.max_len
            while this_seq_len > 0:
                timing_signal.append(self.position_table[:this_seq_len,:])
                this_seq_len -= self.max_len

        timing_signal = torch.cat(timing_signal, dim=0)
        timing_signal = self.timing_dropout(timing_signal, batch_idxs)

        # Combine the content and timing signals
        if self.partitioned:
            annotations = torch.cat([content_annotations, timing_signal], 1)
        else:
            annotations = content_annotations + timing_signal

        #print(annotations.shape)
        annotations = self.layer_norm(self.dropout(annotations, batch_idxs))
        content_annotations = self.dropout(content_annotations, batch_idxs)

        return annotations, content_annotations, timing_signal, batch_idxs