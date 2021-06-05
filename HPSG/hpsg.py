from dependency import Dependency
from constituency import Constituency, InternalParseNode

from vocab import Vocabulary

DEP_PATH = "../samples/dependency.txt"
CON_PATH = "../samples/constituency.txt"

def hpsg():
    dep = Dependency(DEP_PATH)
    dep.load()

    con = Constituency(CON_PATH, dep.head(), dep.type())
    con.load()

    # [[('NNP', 'Ms.'), ('NNP', 'Haag'), ('VBZ', 'plays'), ('NNP', 'Elianti'), ('.', '.')]]
    return con

def vocabularies():
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

    tag_vocab.freeze()
    word_vocab.freeze()
    label_vocab.freeze()
    char_vocab.freeze()
    type_vocab.freeze()

def main():
    vocabularies()

if __name__ == "__main__":
    main()
