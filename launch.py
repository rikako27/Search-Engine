from configparser import ConfigParser
from argparse import ArgumentParser

from utils.config import Config
from utils.indexer import Indexer

def main(config_file, restart, path):
    cparser = ConfigParser()
    cparser.read(config_file)
    config = Config(cparser)
    indexer = Indexer(config, restart, path)
    indexer.create_indexer()
    indexer.create_index_table()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--restart", action="store_true", default=False)
    parser.add_argument("--config_file", type=str, default="config.ini")
    parser.add_argument("--path", type=str, default="DEV")
    args = parser.parse_args()
    main(args.config_file, args.restart, args.path)
