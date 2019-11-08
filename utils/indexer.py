from pathlib import Path
from utils.tokenizer import Tokenizer
import os
import sys
import json
from collections import defaultdict
from math import log
import pickle

class Indexer:
    def __init__(self, config, restart, path):
        self.config = config
        self.path_to_db = Path(path)

        if restart:
            if os.path.exists(self.config.save_hash_doc):
                os.remove(self.config.save_path)

            if os.path.exists(self.config.save_index_table):
                os.remove(self.config.save_index_table)

            if os.path.exists(self.config.save_result):
                os.remove(self.config.save_result)

        self.save_tf = defaultdict(lambda: defaultdict(float))
        self.save_pos = defaultdict(lambda: defaultdict(list))
        self.hash_doc = defaultdict(str)
        self.count_files = 0

    def create_indexer(self):
        for dir in self.path_to_db.iterdir():
            if dir.is_dir():
                str_dir = str(dir)
                for file in dir.iterdir():
                    if not file.is_file():
                        continue
                    str_file = str(file)
                    with open(file, 'r') as file:
                        parsed_json = json.load(file)
                        url = parsed_json['url']
                        content = parsed_json['content']

                    self.hash_doc[self.count_files] = str_file
                    tokenize = Tokenizer(content)
                    tokenize.extract_texts()
                    tf = tokenize.calculate_tf()
                    position_index = tokenize.get_positional_index()
                    for word, score in tf.items():
                        self.save_tf[word][self.count_files] = score
                    for word, index in position_index.items():
                        self.save_pos[word][self.count_files].extend(index)
                    self.count_files += 1

        with open(self.config.save_hash_doc, "w+") as f:
            json.dump(self.hash_doc, f)


    def create_index_table(self):
        for word, file_score in self.save_tf.items():
            idf = log(1.0 * self.count_files / len(self.save_tf[word]))
            for file, score in file_score.items():
                tf_idf = idf * score
                self.save_tf[word][file] = tf_idf

        with open(self.config.save_result, "w+") as result:
            # the number of documents
            result.write("the number of documents %d\n" % self.count_files)
            # the number of [unique] tokens
            result.write("the number of unique tokens %d\n" % len(self.save_tf))

        with open(self.config.save_index_table, "w+") as sace_tf_idf:
            json.dump(self.save_tf, save_tf_idf)
