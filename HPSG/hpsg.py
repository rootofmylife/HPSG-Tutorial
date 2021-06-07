import torch
import numpy as np
import itertools
import time

from dependency import Dependency
from constituency import Constituency, InternalParseNode

from vocab import Vocabulary

from model.chart_parser import ChartParser

DEP_PATH = "../samples/dep_train.txt"
CON_PATH = "../samples/con_train.txt"

def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string

def hpsg():
    dep = Dependency(DEP_PATH)
    dep.load()

    con = Constituency(CON_PATH, dep.head(), dep.type())
    con.load()

    # [[('NNP', 'Ms.'), ('NNP', 'Haag'), ('VBZ', 'plays'), ('NNP', 'Elianti'), ('.', '.')]]
    return con

def train():
    parse_tree = hpsg().get_parse_tree()

    print("Constructing vocabularies...\n")
    tag_vocab = Vocabulary()
    tag_vocab.index("<START>")
    tag_vocab.index("<STOP>")
    tag_vocab.index("UNK")

    word_vocab = Vocabulary()
    word_vocab.index("<START>")
    word_vocab.index("<STOP>")
    word_vocab.index("<UNK>")

    label_vocab = Vocabulary()
    label_vocab.index(())
    label_vocab.index(('<H>',))

    type_vocab = Vocabulary()

    char_vocab = Vocabulary()

    char_set = set()

    for tree in parse_tree:
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, InternalParseNode):
                label_vocab.index(node.label)
                if node.type != "<START>":  # not include root type
                    type_vocab.index(node.type)
                nodes.extend(reversed(node.children))
            else:
                tag_vocab.index(node.tag)
                word_vocab.index(node.word)
                type_vocab.index(node.type)
                char_set |= set(node.word)

    highest_codepoint = max(ord(char) for char in char_set)
    if highest_codepoint < 512:
        if highest_codepoint < 256:
            highest_codepoint = 256
        else:
            highest_codepoint = 512

        for codepoint in range(highest_codepoint):
            char_index = char_vocab.index(chr(codepoint))
            assert char_index == codepoint
    else:
        char_vocab.index("\0")
        char_vocab.index("\1")
        char_vocab.index("\2")
        char_vocab.index("\3")
        char_vocab.index("\4")
        for char in sorted(char_set):
            char_vocab.index(char)

    tag_vocab.freeze() # ['<START>', '<STOP>', 'UNK', 'DT', 'NN', 'JJ', 'VBD', 'CD', 'NNS', 'IN', 'NNP', 'VBZ', '.']
    word_vocab.freeze() # ['<START>', '<STOP>', '<UNK>', 'The', 'luxury', 'auto', 'maker', 'last', 'year', 'sold', '1,214', 'cars', 'in', 'the', 'U.S.']
    label_vocab.freeze() # [(), ('<H>',), ('S',), ('NP',), ('VP',), ('PP',)]
    char_vocab.freeze() # ['\x00', '\x01', '\x02', '\x03', '\x04', .. ]
    type_vocab.freeze() # ['nsubj', 'det', 'nn']

    print("Initializing model...\n")
    parser = ChartParser(tag_vocab, word_vocab, label_vocab, char_vocab, type_vocab)

    print("Initializing optimizer...\n")
    trainable_parameters = [param for param in parser.parameters() if param.requires_grad]
    trainer = torch.optim.Adam(trainable_parameters, lr=1., betas=(0.9, 0.98), eps=1e-9)
    warmup_coeff = 0.0008 / 160
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer, 'max',
        factor=0.5,
        patience=5,
        verbose=True,
    )
    grad_clip_threshold = np.inf

    print("Training model...\n")
    total_processed = 0
    current_processed = 0
    check_every = len(parse_tree) / 4
    best_dev_score = -np.inf
    # best_model_path = None

    start_time = time.time()

    for epoch in itertools.count(start=1):
        if epoch > 2: # Set epoch
            break

        np.random.shuffle(parse_tree)
        epoch_start_time = time.time()

        for start_index in range(0, len(parse_tree), 250):
            trainer.zero_grad()

            iteration = total_processed // 250 + 1
            if iteration <= 160:
                for param_group in trainer.param_groups:
                    param_group['lr'] = iteration * warmup_coeff

            parser.train()

            batch_loss_value = 0.0
            batch_trees = parse_tree[start_index:start_index + 250]

            batch_sentences = [[(leaf.tag, leaf.word) for leaf in tree.leaves()] for tree in batch_trees]
            for subbatch_sentences, subbatch_trees in parser.split_batch(batch_sentences, batch_trees, 2000):
                _, loss = parser.parse_batch(subbatch_sentences, subbatch_trees)

                loss = loss / len(batch_trees)
                loss_value = float(loss.data.cpu().numpy())
                batch_loss_value += loss_value
                if loss_value > 0:
                    loss.backward()
                del loss
                total_processed += len(subbatch_trees)
                current_processed += len(subbatch_trees)

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, grad_clip_threshold)

            trainer.step()

            print(
                "epoch {:,} "
                "batch {:,}/{:,} "
                "processed {:,} "
                "batch-loss {:.4f} "
                "grad-norm {:.4f} "
                "epoch-elapsed {} "
                "total-elapsed {}".format(
                    epoch,
                    start_index // 250 + 1,
                    int(np.ceil(len(parse_tree) / 250)),
                    total_processed,
                    batch_loss_value,
                    grad_norm,
                    format_elapsed(epoch_start_time),
                    format_elapsed(start_time),
                )
            )

            if current_processed >= check_every:
                current_processed -= check_every
                # check_dev(epoch)

        # adjust learning rate at the end of an epoch
        if (total_processed // 250 + 1) > 160:
            scheduler.step(best_dev_score)

    print("Finish training...")

def main():
    train()
    
if __name__ == "__main__":
    main()
