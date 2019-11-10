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
        self.count_tokens = 0
        self.importance = {'title': 6.0, 'h1': 5.0, 'h2': 4.0, 'h3': 3.0, 'b': 2.0}

        self.data = defaultdict(list) #This stores token-id and tf-idf
        self.token = {} #Stores token and token-id

        for digit in range(0, 10):
            self.token[str(digit)] = defaultdict(int)
        for alpha in string.ascii_lowercase:
            self.token[alpha] = defaultdict(int)

        self.data_store = "database"
        if not os.path.exists(self.data_store):
            os.makedirs(self.data_store)

        for key in self.data.keys():
            file = self.data_store + "/" + str(key)
            if os.path.exists(file):
                os.remove(file)

        # Create term_ID files
        self.doc_file = self.data_store + "/" + "doc_id.txt"
        self.token_files = [self.data_store + "/" + "token_" + str(k) + ".txt" for k in self.token.keys()]
        self.index_file = self.data_store + "/" + "index.pkl"

    def add_tokens_to_dictionary(self, tokenid_tf: dict, doc_id: int):
        for id, tf in tokenid_tf.items():
            posting = Posting(doc_id, id, tf)
            self.data[id].append(posting)

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
                    tokenid_tf = self.map_from_token_to_id(token_tf)
                    self.add_tokens_to_dictionary(tokenid_tf, self.count_files)
                    self.count_files += 1
        hash_doc.close()
        self.recalculate_tf_idf()
        #save token and token_id to text file
        self.save_tokenid()
        self.save_to_file()


    def map_from_token_to_id(self, token_tf):
        mapped_dict = defaultdict(float)
        for key, val in token_tf.items():
            if key not in self.token[key[0]].keys():
                self.token[key[0]][key] = self.count_tokens
                self.count_tokens += 1
            mapped_dict[self.token[key[0]][key]] = val
        return mapped_dict

    def save_to_file(self):
        with open(self.index_file, "wb") as write_file:
            pickle.dump(self.data, write_file)

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
        self.data.clear()
        self.data = defaultdict(list)

        print("the number of documents %d\n" % self.count_files)

    def recalculate_tf_idf(self):
        for token, list_posting in self.data.items():
            num_doc = len(list_doc_tf)
            for posting in list_posting:
                posting.calculate_tfidf(self.count_files, num_doc)

        print("Number of all unique tokens: %d\n" % self.count_tokens)

    def save_tokenid(self):
        token_fd = [open(fd, "w+") for fd in self.token_files]
        for key, token_id in self.token.items():
            if key.isdigit():
                for token, id in token_id.items():
                    token_fd[int(key)].write("%s, %d\n" % (token, id))
            else:
                index = ord(key) - 87
                for token, id in token_id.items():
                    token_fd[index].write("%s, %d\n" % (token, id))
        for fd in token_fd:
            fd.close()
