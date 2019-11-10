from configparser import ConfigParser
from argparse import ArgumentParser
from utils.indexer import Indexer

def main(path):
    indexer = Indexer(path)
    indexer.create_indexer()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="/home/lopes/Datasets/IR/DEV")
    #parser.add_argument("--path", type=str, default="../DEV")
    
    args = parser.parse_args()
    main(args.path)
