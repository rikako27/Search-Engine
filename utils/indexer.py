from pathlib import Path
from utils.tokenizer import Tokenizer
from collections import defaultdict
import os
import sys
import json
from math import log
import string
import gc

class Indexer:
    def __init__(self, path):
        self.path_to_db = Path(path)
        self.hash_doc = defaultdict(str)
        self.count_files = 0
        self.importance = {'title': 6.0, 'h1': 5.0, 'h2': 4.0, 'h3': 3.0, 'b': 2.0}
        self.data = {}
        for alpha in string.ascii_lowercase:
            self.data[alpha] = defaultdict(list)
        for num in range(0, 10):
            self.data[str(num)] = defaultdict(list)

        self.data_store = "DATA"
        if not os.path.exists(self.data_store):
            os.makedirs(self.data_store)

        for key in self.data.keys():
            file = self.data_store + "/" + str(key)
            if os.path.exists(file):
                os.remove(file)

    def add_tokens_to_dictionary(self, tokens_tf: dict, doc_id: int):
        for token, tf in tokens_tf.items():
            self.data[token[0]][token].append([doc_id, tf])

        if self.count_files != 0 and self.count_files % 200 == 0:
            print("Now appending")
            self.save_to_file()

    def removeFragment(self, url:str):
        return url.split("#")[0]

    def create_indexer(self):
        #hash_doc file stores mapping of doc_id and url
        hash_doc = open("doc_id.txt", "w+")
        for dir in self.path_to_db.iterdir():
            if dir.is_dir():
                for file in dir.iterdir():
                    gc.collect()
                    if not file.is_file():
                        continue
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
        self.save_to_file()
        self.recalculate_tf_idf()

    def save_to_file(self):
        for key, tf_idf in self.data.items():
            file = self.data_store + "/" + str(key)
            if not os.path.exists(file):
                with open(file, "w") as write_file:
                    json.dump(tf_idf, write_file)
            else: #need merging
                with open(file, "r") as original: #grap the saved dictionary
                    first_dict = json.load(original)
                #Now appending
                for k, v in first_dict.items():
                    v.extend(tf_idf[k])
                with open(file, "a") as write_file:
                    json.dump(first_dict, write_file)

        #make dictionary empty
        for key in self.data.keys():
            self.data[key] = defaultdict(list)

        print("the number of documents %d\n" % self.count_files)

    def recalculate_tf_idf(self):
        for key in self.data.keys():
            file = self.data_store + "/" + str(key)

            with open(file, "r") as read_index:
                dict = json.load(read_index)
                for token, list_doc_tf in dict.items():
                    num_doc = len(list_doc_tf)
                    for index, id_tf in enumerate(list_doc_tf):
                        idf = log(1.0 * self.count_files / num_doc)
                        tf_idf = idf * id_tf[1]
                        dict[token][index][1] = round(tf_idf + 0.005, 2)

            print("Number of unique tokens in folder %s: %d\n" % (key, len(dict)))
            #update dict
            with open(file, "w") as write_file:
                json.dump(dict, write_file)
