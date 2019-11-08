from configparser import ConfigParser
from argparse import ArgumentParser

from utils.indexer import Indexer

def main(path):
    indexer = Indexer(path)
    indexer.create_indexer()
    indexer.save_to_file()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="../DEV/aiclub_ics_uci_edu")
    args = parser.parse_args()
    main(args.path)
