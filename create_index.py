from argparse import ArgumentParser
from utils.indexer import Indexer

def main(path):
    indexer1 = Indexer(path, 1)
    indexer1.create_indexer()

    # indexer2 = Indexer(path, 2)
    # indexer2.create_indexer()
    #
    # indexer3 = Indexer(path, 3)
    # indexer3.create_indexer()

if __name__ == "__main__":
    parser = ArgumentParser()
    #parser.add_argument("--path", type=str, default="/home/lopes/Datasets/IR/DEV")
    parser.add_argument("--path", type=str, default="../DEV")
    args = parser.parse_args()
    main(args.path)
