from pathlib import Path
import os
import sys
import json
from collections import defaultdict
from math import log
import string
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, RegexpTokenizer
import re
import gc

class Indexer:
    def __init__(self, path):
        self.path_to_db = Path(path)
        self.hash_doc = defaultdict(str)
        self.count_files = 0
        self.importance = {'title': 6.0, 'h1': 5.0, 'h2': 4.0, 'h3': 3.0, 'b': 2.0}
        self.data = {}
        self.visited_url = set()
        for alpha in string.ascii_lowercase:
            self.data[alpha] = defaultdict(list)
        for num in range(0, 10):
            self.data[str(num)] = defaultdict(list)

        self.data_store = "DATA"
        self.save_docID = "doc_id.json"

    def tokens(self, text: str) -> list:
        #split text into all alphanumeric tokens and perform porter stemming
        tokenizer = RegexpTokenizer('[0-9A-Za-z]+', flags=re.UNICODE)
        stemmer = PorterStemmer()
        return [stemmer.stem(t).lower() for t in tokenizer.tokenize(text)]

    def calculate_tf(self, word_freq: dict):
        tf = defaultdict(float)
        total_words = sum(word_freq.values())
        for word, freq in word_freq.items():
            tf[word] = freq / total_words
        return tf

    def extract_texts(self, html_text: str):
        word_freq = defaultdict(float)
        bs = BeautifulSoup(html_text, 'lxml')

        #delete <script> and <style> elements
        for script in bs(["script", "style"]):
            script.extract()

        #find header, bold, and title, and add weights
        for text in bs.find_all(['title', 'h1', 'h2', 'h3', 'b']):
            tag_name = text.name
            list_tokens = self.tokens(text.get_text().lower())
            for token in list_tokens:
                word_freq[token] += self.importance[tag_name]

        all_texts = self.tokens(bs.get_text().lower())
        for text in all_texts:
            word_freq[text] += 1.0

        return self.calculate_tf(word_freq)

    def add_tokens_to_dictionary(self, tokens_tf: dict, doc_id: int):
        for token, tf in tokens_tf.items():
            self.data[token[0]][token].append([doc_id, tf])

    def recalculate_tf_idf(self):
        for index, file_score in self.data.items():
            for token, l in file_score.items():
                for i, elem in enumerate(l):
                    num_doc = len(self.data[index][token])
                    idf = log(1.0 * self.count_files / num_doc)
                    tf_idf = idf * elem[1]
                    self.data[index][token][i][1] = tf_idf

    def removeFragment(self, url:str):
        return url.split("#")[0]

    def create_indexer(self):
        hash_doc = open("doc_id.txt", "w+")
        for dir in self.path_to_db.iterdir():
            if dir.is_dir():
                str_dir = str(dir)
                for file in dir.iterdir():
                    if not file.is_file():
                        continue
                    str_file = str(file)
           
                    with open(file, 'r', encoding="utf8", errors="ignore") as file:
                        parsed_json = json.load(file)
                        url = self.removeFragment(parsed_json['url'])
                        if url in self.visited_url:
                            continue
                        self.visited_url.add(url)
                        content = parsed_json['content']

                    #self.hash_doc[self.count_files] = url
                    hash_doc.write("%d, %s\n" % (self.count_files, url))
                    token_tf = self.extract_texts(content)
                    self.add_tokens_to_dictionary(token_tf, self.count_files)
                    self.count_files += 1
        self.recalculate_tf_idf()
        hash_doc.close()

    def save_to_file(self):
        if not os.path.exists(self.data_store):
            os.makedirs(self.data_store)

        with open(self.save_docID, "w+") as f:
            json.dump(self.hash_doc, f)

        for key, tf_idf in self.data.items():
            if os.path.exists(self.data_store + "/" + str(key)):
                os.remove(self.data_store + "/" + str(key))
            with open(self.data_store + "/" + str(key), "w") as write_file:
                write_file.write(str(tf_idf))

        print("the number of documents %d\n" % self.count_files)
        num_unique_tokens = sum(len(i.values()) for i in self.data.values())
        print("the number of unique tokens %d\n" % num_unique_tokens)
