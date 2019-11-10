from pathlib import Path
from utils.tokenizer import Tokenizer
from utils.posting import Posting
from collections import defaultdict
import os
import sys
import json
from math import log
import string
import gc
import pickle

class Indexer:
    def __init__(self, path):
        self.path_to_db = Path(path)
        self.hash_doc = defaultdict(str)
        self.count_files = 0
        self.importance = {'title': 6.0, 'h1': 5.0, 'h2': 4.0, 'h3': 3.0, 'b': 2.0}

        self.data = {} #This stores token-id and tf-idf

        for alpha in string.ascii_lowercase:
            self.data[alpha] = defaultdict(list)
        for digit in range(0, 10):
            self.data[str(digit)] = defaultdict(list)

        self.data_store = "database"
        if not os.path.exists(self.data_store):
            os.makedirs(self.data_store)

        for key in self.data.keys():
            file = self.data_store + "/" + str(key)
            if os.path.exists(file):
                os.remove(file)

        # Create term_ID files
        self.doc_file = self.data_store + "/" + "doc_id.txt"

    def add_tokens_to_dictionary(self, token_tf: dict, doc_id: int):
        for t, tf in token_tf.items():
            posting = Posting(doc_id, t, tf)
            self.data[t[0]][t].append(posting)

    def removeFragment(self, url:str):
        return url.split("#")[0]

    def create_indexer(self):
        #hash_doc file stores mapping of doc_id and url
        hash_doc = open(self.doc_file, "w+")
        for dir in self.path_to_db.iterdir():
            if dir.is_dir():
                for file in dir.iterdir():
                    gc.collect()
                    if not file.is_file():
                        continue
        #for file in self.path_to_db.iterdir():
                    with open(file, 'r', encoding="ascii", errors="ignore") as file:
                        parsed_json = json.load(file)
                        url = self.removeFragment(parsed_json['url'])
                        content = parsed_json['content']

                    hash_doc.write("%d, %s\n" % (self.count_files, url))
                    tokenizer = Tokenizer(content)
                    token_tf = tokenizer.extract_texts()
                    self.add_tokens_to_dictionary(token_tf, self.count_files)
                    self.count_files += 1
        hash_doc.close()
        self.recalculate_tf_idf()
        self.save_to_file()

    def save_to_file(self):
        for key in self.data.keys():
            files = self.data_store + "/" + key

            with open(file, "wb") as write_file:
                pickle.dump(self.data[key], write_file)

        # if not os.path.exists(self.data_file):
        #     with open(self.data_file, "wb") as write_file:
        #         pickle.dump(self.data, write_file)
        # else: #need merging
        #     with open(self.data_file, "rb") as original: #grap the saved dictionary
        #         first_dict = pickle.load(original)
        #     #Now appending
        #     for k, v in first_dict.items():
        #         v.extend(self.data[k])
        #     with open(self.data_file, "ab") as write_file:
        #         pickle.dump(first_dict, write_file)

        #make dictionary empty
        # self.data.clear()
        # self.data = defaultdict(list)

        print("the number of documents %d\n" % self.count_files)

    def recalculate_tf_idf(self):
        for key in self.data.keys():
            for token, posting in self.data[key].items():
                num_doc = len(posting)
                for posting in list_posting:
                    posting.calculate_tfidf(self.count_files, num_doc)
