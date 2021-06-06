import torch.nn as nn

from model.multihead_attention import MultiHeadAttention
from model.label_attention import LabelAttention
from model.positionwise_feedforward import PositionwiseFeedForward
from model.partitioned_positionwise_feedforward import PartitionedPositionwiseFeedForward

class Encoder(nn.Module):
    def __init__(self, embedding,
                    num_layers=1, num_heads=2, d_kv = 32, d_ff=1024, d_l=112,
                    d_positional=None,
                    num_layers_position_only=0,
                    relu_dropout=0.1, residual_dropout=0.1, attention_dropout=0.1,
                    use_lal=True,
                    lal_d_kv=128,
                    lal_d_proj=128,
                    lal_resdrop=True,
                    lal_pwff=True,
                    lal_q_as_matrix=False,
                    lal_partitioned=True):
        super().__init__()
        self.embedding_container = [embedding]
        d_model = embedding.d_embedding

        d_k = d_v = d_kv

        self.stacks = []

        for i in range(12):
            attn = MultiHeadAttention(num_heads, d_model, d_k, d_v, residual_dropout=residual_dropout,
                                      attention_dropout=attention_dropout, d_positional=d_positional)
            if d_positional is None:
                ff = PositionwiseFeedForward(d_model, d_ff, relu_dropout=relu_dropout,
                                             residual_dropout=residual_dropout)
            else:
                ff = PartitionedPositionwiseFeedForward(d_model, d_ff, d_positional, relu_dropout=relu_dropout,
                                                        residual_dropout=residual_dropout)

            self.add_module(f"attn_{i}", attn)
            self.add_module(f"ff_{i}", ff)

            self.stacks.append((attn, ff))

        if use_lal:
            lal_d_positional = d_positional if lal_partitioned else None
            attn = LabelAttention(d_model, lal_d_kv, lal_d_kv, d_l, lal_d_proj, use_resdrop=lal_resdrop, q_as_matrix=lal_q_as_matrix,
                                  residual_dropout=residual_dropout, attention_dropout=attention_dropout, d_positional=lal_d_positional)
            ff_dim = lal_d_proj * d_l
            # if hparams.lal_combine_as_self:
            #     ff_dim = d_model
            if lal_pwff:
                if d_positional is None or not lal_partitioned:
                    ff = PositionwiseFeedForward(ff_dim, d_ff, relu_dropout=relu_dropout, residual_dropout=residual_dropout)
                else:
                    ff = PartitionedPositionwiseFeedForward(ff_dim, d_ff, d_positional, relu_dropout=relu_dropout, residual_dropout=residual_dropout)
            else:
                ff = None

            self.add_module(f"attn_{num_layers}", attn)
            self.add_module(f"ff_{num_layers}", ff)
            self.stacks.append((attn, ff))

        self.num_layers_position_only = num_layers_position_only
        if self.num_layers_position_only > 0:
            assert d_positional is None, "num_layers_position_only and partitioned are incompatible"

    def forward(self, xs, pre_words_idxs, batch_idxs, extra_content_annotations=None):
        emb = self.embedding_container[0]
        res, res_c, timing_signal, batch_idxs = emb(xs, pre_words_idxs, batch_idxs, extra_content_annotations=extra_content_annotations)

        for i, (attn, ff) in enumerate(self.stacks):
            res, current_attns = attn(res, batch_idxs)
            if ff is not None:
                res = ff(res, batch_idxs)

        return res, current_attns