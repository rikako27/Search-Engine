from pathlib import Path
from utils.tokenizer import Tokenizer
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
        self.count_files = 0
        
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
            self.data[t[0]][t].append((doc_id, tf))

        if self.count_files != 0 and self.count_files % 300 == 0:
            self.save_to_file()

    def removeFragment(self, url:str):
        return url.split("#")[0]

    def create_indexer(self):
        #hash_doc file stores mapping of doc_id and url
        hash_doc = open(self.doc_file, "w+")
        for dir in self.path_to_db.iterdir():
            if dir.is_dir():
                for file in dir.iterdir():
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
                    gc.collect() 
                    self.count_files += 1
        hash_doc.close()
        self.save_to_file()
        self.recalculate_tf_idf()
       

    def save_to_file(self):
        for key in self.data.keys():
            file = self.data_store + "/" + key

            if not os.path.exists(file):
                with open(file, "wb") as write_file:
                    pickle.dump(self.data[key], write_file,  protocol=pickle.HIGHEST_PROTOCOL)
                self.data[key].clear()
                self.data[key] = defaultdict(list)

            else: #need merging
                with open(file, "rb") as f: #grap the saved dictionary
                    orig_dict = pickle.load(f)

                #Now appending
                for token, posting in self.data[key].items():
                    if token not in orig_dict.keys():
                        orig_dict[token] = list()
                    orig_dict[token].append(posting)
                self.data[key].clear()
                self.data[key] = defaultdict(list)

                with open(file, "wb") as write_file:
                    pickle.dump(orig_dict, write_file, protocol=pickle.HIGHEST_PROTOCOL)
                orig_dict = {} 
            gc.collect()
        print("the number of documents %d\n" % self.count_files)

    def recalculate_tf_idf(self):
        for key in self.data.keys():
            file = self.data_store + "/" + key 
            with open(file, "rb") as rf:
                token_posting = pickle.load(rf)
            os.remove(file)
            for token, posting in token_posting.items():
                num_doc = len(posting)
                for index, elem in enumerate(posting):
                    idf = log(1.0 * self.count_files / num_doc)
                    tf_idf = idf * elem[1]
                    token_posting[token][index][1] = round(tf_idf + 0.005, 2)
            with open(file, "wb") as wf:
                pickle.dump(token_posting, wf, protocol=pickle.HIGHEST_PROTOCOL)
            gc.collect()
