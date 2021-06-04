from dependency import Dependency
from constituency import Constituency

DEP_PATH = "../samples/dependency.txt"
CON_PATH = "../samples/constituency.txt"

def hpsg():
    dep = Dependency(DEP_PATH)
    dep.load()

    con = Constituency(CON_PATH, dep.head(), dep.type())
    con.load()

    # [[('NNP', 'Ms.'), ('NNP', 'Haag'), ('VBZ', 'plays'), ('NNP', 'Elianti'), ('.', '.')]]
    return con.get_hpsg_tree()


if __name__ == "__main__":
    hpsg()
