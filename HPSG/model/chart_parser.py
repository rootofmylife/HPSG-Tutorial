import torch
import torch.nn as nn
from torch import from_numpy

import numpy as np

from model.multilevel_embedding import MultiLevelEmbedding
from model.encoder import Encoder
from model.layer_normalization import LayerNormalization
from model.dep_score import DepScore
# from ..constituency import InternalParseNode, LeafParseNode

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})

import model.hpsg_decoder
import model.const_decoder

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"
TAG_UNK = "UNK"

Sub_Head = "<H>"

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

    def split_batch(self, sentences, golds, subbatch_max_tokens=3000):
        lens = [len(sentence) + 2 for sentence in sentences]

        lens = np.asarray(lens, dtype=int)
        lens_argsort = np.argsort(lens).tolist()

        num_subbatches = 0
        subbatch_size = 1
        while lens_argsort:
            if (subbatch_size == len(lens_argsort)) or (subbatch_size * lens[lens_argsort[subbatch_size]] > subbatch_max_tokens):
                yield [sentences[i] for i in lens_argsort[:subbatch_size]], [golds[i] for i in lens_argsort[:subbatch_size]]
                lens_argsort = lens_argsort[subbatch_size:]
                num_subbatches += 1
                subbatch_size = 1
            else:
                subbatch_size += 1

    def parse_batch(self, sentences, golds=None):
        is_train = golds is not None
        self.train(is_train)
        torch.set_grad_enabled(True)

        if golds is None:
            golds = [None] * len(sentences)

        packed_len = sum([(len(sentence) + 2) for sentence in sentences])

        i = 0
        tag_idxs = np.zeros(packed_len, dtype=int)
        word_idxs = np.zeros(packed_len, dtype=int)
        batch_idxs = np.zeros(packed_len, dtype=int)
        for snum, sentence in enumerate(sentences):
            for (tag, word) in [(START, START)] + sentence + [(STOP, STOP)]:
                tag_idxs[i] = 0 if (not self.use_tags) else self.tag_vocab.index_or_unk(tag, TAG_UNK)
                if word not in (START, STOP):
                    count = self.word_vocab.count(word)
                    if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                        word = UNK
                word_idxs[i] = self.word_vocab.index(word)
                batch_idxs[i] = snum
                i += 1

        batch_idxs = BatchIndices(batch_idxs)

        emb_idxs_map = {
            'tags': tag_idxs,
            'words': word_idxs,
        }
        emb_idxs = [
            from_numpy(emb_idxs_map[emb_type]).requires_grad_(False)
            for emb_type in self.emb_types
        ]
        pre_words_idxs = from_numpy(word_idxs).requires_grad_(False)

        extra_content_annotations_list = []
        extra_content_annotations = None

        if len(extra_content_annotations_list) > 1 :
            extra_content_annotations = sum(extra_content_annotations_list)
        elif len(extra_content_annotations_list) == 1:
            extra_content_annotations = extra_content_annotations_list[0]

        annotations, self.current_attns = self.encoder(emb_idxs, pre_words_idxs, batch_idxs, extra_content_annotations=extra_content_annotations)

        if self.partitioned and not self.use_lal:
            annotations = torch.cat([
                annotations[:, 0::2],
                annotations[:, 1::2],
            ], 1)

        if self.use_lal and not self.lal_combine_as_self:
            half_dim = self.lal_d_proj//2
            d_l = self.label_vocab.size - 1
            fencepost_annotations = torch.cat(
                [annotations[:-1, (i*self.lal_d_proj):(i*self.lal_d_proj + half_dim)] for i in range(d_l)] 
                + [annotations[1:, (i*self.lal_d_proj + half_dim):((i+1)*self.lal_d_proj)] for i in range(d_l)], 1)
        else:
            fencepost_annotations = torch.cat([
                annotations[:-1, :self.d_model//2],
                annotations[1:, self.d_model//2:],
                ], 1)

        fencepost_annotations_start = fencepost_annotations
        fencepost_annotations_end = fencepost_annotations

        fp_startpoints = batch_idxs.boundaries_np[:-1]
        fp_endpoints = batch_idxs.boundaries_np[1:] - 1

        if not is_train:
            trees = []
            scores = []
            for i, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):
                tree, score = self.parse_from_annotations(fencepost_annotations_start[start:end,:],
                                                                        fencepost_annotations_end[start:end,:], sentences[i], i)
                trees.append(tree)
                scores.append(score)

            return trees, scores

        pis = []
        pjs = []
        plabels = []
        paugment_total = 0.0
        cun = 0
        num_p = 0
        gis = []
        gjs = []
        glabels = []
        with torch.no_grad():
            for i, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):

                p_i, p_j, p_label, p_augment, g_i, g_j, g_label \
                    = self.parse_from_annotations(fencepost_annotations_start[start:end,:], fencepost_annotations_end[start:end,:], sentences[i], i, gold=golds[i])

                paugment_total += p_augment
                num_p += p_i.shape[0]
                pis.append(p_i + start)
                pjs.append(p_j + start)
                gis.append(g_i + start)
                gjs.append(g_j + start)
                plabels.append(p_label)
                glabels.append(g_label)

        cells_i = from_numpy(np.concatenate(pis + gis))
        cells_j = from_numpy(np.concatenate(pjs + gjs))
        cells_label = from_numpy(np.concatenate(plabels + glabels))

        cells_label_scores = self.f_label(fencepost_annotations_end[cells_j] - fencepost_annotations_start[cells_i])
        cells_label_scores = torch.cat([
            cells_label_scores.new_zeros((cells_label_scores.size(0), 1)),
            cells_label_scores
        ], 1)
        cells_label_scores = torch.gather(cells_label_scores, 1, cells_label[:, None])
        loss = cells_label_scores[:num_p].sum() - cells_label_scores[num_p:].sum() + paugment_total

        cun = 0
        for snum, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):
            leng = end - start
            arc_score, type_score = self.dep_score(fencepost_annotations_start[start:end,:], fencepost_annotations_end[start:end,:])
            arc_gather = [leaf.father for leaf in golds[snum].leaves()]
            type_gather = [self.type_vocab.index(leaf.type) for leaf in golds[snum].leaves()]
            cun += 1
            assert len(arc_gather) == leng - 1
            arc_score = torch.transpose(arc_score,0, 1)
            loss = loss + 0.5 * self.loss_func(arc_score[1:, :], from_numpy(np.array(arc_gather)).requires_grad_(False)) \
                   + 0.5 * self.loss_funt(type_score[1:, :],from_numpy(np.array(type_gather)).requires_grad_(False))

        return None, loss

    def parse_from_annotations(self, fencepost_annotations_start, fencepost_annotations_end, sentence, sentence_idx, gold=None):
        is_train = gold is not None
        contributions = None
        label_scores_chart = self.label_scores_from_annotations(fencepost_annotations_start, fencepost_annotations_end)
        label_scores_chart_np = label_scores_chart.cpu().data.numpy()

        if is_train:
            decoder_args = dict(
                sentence_len=len(sentence),
                label_scores_chart=label_scores_chart_np,
                gold=gold,
                label_vocab=self.label_vocab,
                is_train=is_train)

            p_score, p_i, p_j, p_label, p_augment = const_decoder.decode(False, **decoder_args)
            g_score, g_i, g_j, g_label, g_augment = const_decoder.decode(True, **decoder_args)
            return p_i, p_j, p_label, p_augment, g_i, g_j, g_label
        else:
            arc_score, type_score = self.dep_score(fencepost_annotations_start, fencepost_annotations_end)

            arc_score_dc = torch.transpose(arc_score, 0, 1)
            arc_dc_np = arc_score_dc.cpu().data.numpy()

            type_np = type_score.cpu().data.numpy()
            type_np = type_np[1:, :]  # remove root
            type = type_np.argmax(axis=1)
            return self.decode_from_chart(sentence, label_scores_chart_np, arc_dc_np, type, sentence_idx=sentence_idx, contributions=contributions)

    def decode_from_chart(self, sentence, label_scores_chart_np, arc_dc_np, type, sentence_idx=None, gold=None, contributions=None):

        decoder_args = dict(
            sentence_len=len(sentence),
            label_scores_chart= label_scores_chart_np * 0.5,
            type_scores_chart = arc_dc_np * (1.0 - 0.5),
            gold=gold,
            label_vocab=self.label_vocab,
            type_vocab = self.type_vocab,
            is_train=False)

        force_gold = (gold is not None)

        score, p_i, p_j, p_label, p_father, p_type, _ = hpsg_decoder.decode(force_gold, **decoder_args)
        if contributions is not None:
            d_l = (self.label_vocab.size - 2)
            mb_size = (self.current_attns.shape[0] // d_l)
            print('SENTENCE', sentence)

        idx = -1
        def make_tree():
            nonlocal idx
            idx += 1
            i, j, label_idx = p_i[idx], p_j[idx], p_label[idx]
            label = self.label_vocab.value(label_idx)
            if contributions is not None:
                if label_idx > 0:
                    print(i, sentence[i], j, sentence[j-1], label, label_idx, contributions[i,j,label_idx-1])
                    print("CONTRIBUTIONS")
                    print(list(enumerate(contributions[i,j])))
                    print("ATTENTION DIST")
                    print(torch.softmax(self.current_attns[sentence_idx::mb_size, 0, i:j+1], -1))
            if (i + 1) >= j:
                tag, word = sentence[i]
                if type is not None:
                    tree = LeafParseNode(int(i), tag, word, p_father[i], self.type_vocab.value(type[i]))
                else:
                    tree = LeafParseNode(int(i), tag, word, p_father[i], self.type_vocab.value(p_type[i]))
                if label:
                    assert label[0] != Sub_Head
                    tree = InternalParseNode(label, [tree])
                return [tree]
            else:
                left_trees = make_tree()
                right_trees = make_tree()
                children = left_trees + right_trees
                if label and label[0] != Sub_Head:
                    return [InternalParseNode(label, children)]
                else:
                    return children

        tree_list = make_tree()
        assert len(tree_list) == 1
        tree = tree_list[0]
        return tree, score

    def label_scores_from_annotations(self, fencepost_annotations_start, fencepost_annotations_end):

        span_features = (torch.unsqueeze(fencepost_annotations_end, 0)
                         - torch.unsqueeze(fencepost_annotations_start, 1))

        label_scores_chart = self.f_label(span_features)
        label_scores_chart = torch.cat([
            label_scores_chart.new_zeros((label_scores_chart.size(0), label_scores_chart.size(1), 1)),
            label_scores_chart
            ], 2)
        return label_scores_chart

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
