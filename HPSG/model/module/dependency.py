
class Dependency(object):
    def __init__(self, file_path) -> None:
        super().__init__()
        self._file_path = file_path
        self.word_list = []
        self.pos_list = []
        self.gold_pos_list = []
        self.head_list = []
        self.type_list = []

    def load(self):
        with open(self._file_path) as fin:
            lines = [line.strip() for line in fin]

        # Fix bug that not includes final word.
        lines.append(None)

        i, start = 0, 0

        for line in lines:
            if not line or i == (len(lines) - 1):
                w, p, gp, h, t = self.process_sentence(lines[start:i])
                self.word_list.append(w)
                self.pos_list.append(p)
                self.gold_pos_list.append(gp)
                self.head_list.append(h)
                self.type_list.append(t)
                start = i + 1
            i += 1

    def process_sentence(self, lines):
        word = []
        pos = []
        gold_pos = []
        head = []
        type = []

        for line in lines:
            value = line.split('\t')
            word.append(value[1])
            pos.append(value[4])
            gold_pos.append(value[3])
            head.append(int(value[6]))
            type.append(value[7])

        return word, pos, gold_pos, head, type

    def word(self):
        return self.word_list

    def pos(self):
        return self.pos_list

    def gold_pos(self):
        return self.gold_pos_list

    def head(self):
        return self.head_list

    def type(self):
        return self.type_list
