from pathlib import Path
from utils.tokenizer import Tokenizer
from collections import defaultdict
import os
import json
from math import log
from string import ascii_lowercase
import shelve
import pickle

class Indexer:
    def __init__(self, path, ngram=3):
        self.path_to_db = Path(path)
        self.count_files = 0
        self.ngram = ngram

        self.data = {} #This stores token-id and tf-idf
        for alpha in ascii_lowercase:
            self.data[alpha] = defaultdict(list)
        for digit in range(0, 10):
            self.data[str(digit)] = defaultdict(list)

        self.data_store = "database" + str(self.ngram)
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
            self.data[t[0]][t].append([doc_id, tf])

        if self.count_files != 0 and self.count_files % 5000 == 0:
            self.save_to_file()

    def removeFragment(self, url:str):
        return url.split("#")[0]

    def create_indexer(self):
        #get the list of ulrs to be removed
        with open('database/removed_urls.pkl', 'rb') as f:
            removed_url = pickle.load(f)

        #hash_doc file stores mapping of doc_id and url
        hash_doc = open(self.doc_file, "w+")
        for dir in self.path_to_db.iterdir():
            if dir.is_dir():
                for file in dir.iterdir():
                    if not file.is_file():
                        continue
                    with open(file, 'r', encoding="ascii", errors="ignore") as file:
                        parsed_json = json.load(file)
                        url = parsed_json['url']
                        if url in removed_url:
                            continue
                        url = self.removeFragment(url)
                        content = parsed_json['content']

                    tokenizer = Tokenizer(content, self.ngram)
                    token_tf = tokenizer.extract_texts()
                    hash_doc.write("%d, %s, %d\n" % (self.count_files, url, tokenizer.length))
                    self.add_tokens_to_dictionary(token_tf, self.count_files)
                    self.count_files += 1

        hash_doc.close()
        self.save_to_file()
        self.recalculate_tf_idf()

    def save_to_file(self):
        for key in self.data.keys():
            file = self.data_store + "/" + key

            if not os.path.exists(file):
                s = shelve.open(file)
                for token in self.data[key]:
                    s[token] = self.data[key][token]
                s.close()
                self.data[key].clear()
                self.data[key] = defaultdict(list)

            else: #need merging
                #grap the saved dictionary
                orig_dict = shelve.open(file, flag="r")
                new_file = file + "_temp"
                new_dict = shelve.open(new_file)

                orig_keys = set(key for key in orig_dict)
                #Now appending
                for token, posting in self.data[key].items():
                    if token in orig_keys:
                        l = orig_dict[token]
                        l.extend(posting)
                        new_dict[token] = l
                    else:
                        new_dict[token] = posting
                keys_only_in_orig = orig_keys - set(self.data[key])
                self.data[key].clear()
                self.data[key] = defaultdict(list)
                for token in keys_only_in_orig:
                    new_dict[token] = orig_dict[token]
                orig_dict.close()
                new_dict.close()

                if os.path.exists(file): #remove original file
                    os.remove(file)

                #rename new_dict
                os.rename(new_file, file)

    def recalculate_tf_idf(self):
        print("Number of Documents %d\n" & self.count_files)
        for key in self.data:
            file = self.data_store + "/" + key
            token_posting = open(file, writeback=True)

            for token, posting in token_posting.items():
                num_doc = len(posting)
                for index, elem in enumerate(posting):
                    idf = log(1.0 * self.count_files / num_doc)
                    tf_idf = idf * (1 + log(elem[1]))
                    token_posting[token][index][1] = round(tf_idf, 8)
            token_posting.close()
