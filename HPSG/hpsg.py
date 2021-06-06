from dependency import Dependency
from constituency import Constituency, InternalParseNode

from vocab import Vocabulary

from model.chart_parser import ChartParser

DEP_PATH = "../samples/dependency.txt"
CON_PATH = "../samples/constituency.txt"

def hpsg():
    dep = Dependency(DEP_PATH)
    dep.load()

    con = Constituency(CON_PATH, dep.head(), dep.type())
    con.load()

    # [[('NNP', 'Ms.'), ('NNP', 'Haag'), ('VBZ', 'plays'), ('NNP', 'Elianti'), ('.', '.')]]
    return con

def train():
    parse_tree = hpsg().get_parse_tree()

    print("Constructing vocabularies...")
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

    print("Initializing model...")
    parser = ChartParser(tag_vocab, word_vocab, label_vocab, char_vocab, type_vocab)

    print("Training model...")

    print("Finish training...")

def main():
    train()
    
if __name__ == "__main__":
    main()
