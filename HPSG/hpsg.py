from dependency import Dependency
from constituency import Constituency

DEP_PATH = "../samples/dependency.txt"
CON_PATH = "../samples/constituency.txt"

def hpsg():
    dep = Dependency(DEP_PATH)
    dep.load()

    # print(dep.word())
    # print(dep.pos())
    # print(dep.gold_pos())
    # print(dep.head())
    # print(dep.type())

    con = Constituency(CON_PATH, dep.head(), dep.type())
    con.load()

    print(con.get_hpsg_tree())

if __name__ == "__main__":
    hpsg()
