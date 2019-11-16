from pathlib import Path
import os
import json
import re
import pickle
import hashlib
from collections import Counter
from bs4 import BeautifulSoup

class Simhash:
    def __init__(self, content, hashbits=128):
        self.content = content
        self.hashbits = hashbits
        self.tokens = self._tokenize()
        self.size = len(self.tokens)
        self.hash = self.simhash()

    #1. Process the document into a set of features with associated weights
    def _tokenize(self):
        bs = BeautifulSoup(self.content, 'lxml')

        #delete <script> and <style> elements
        for script in bs(["script", "style"]):
            script.extract()

        regex = '[a-zA-Z0-9]+'
        tokens = re.findall(regex, bs.get_text().lower())
        return tokens

    def generate_ngrams(self, ngram=3):
        n_grams = zip(*[self.tokens[i:] for i in range(ngram)])
        return Counter([" ".join(n_gram) for n_gram in n_grams])

    def _hashfunc(self, token):
        return int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)

    #Generate a hash value with hashbits for each word
    def simhash(self):
        v = [0] * self.hashbits
        masks = [1 << i for i in range(self.hashbits)]

        ngrams = dict(self.generate_ngrams())
        for token, w in ngrams.items():
            hash = self._hashfunc(token)
            weight = w

            for b in range(self.hashbits):
                if hash & masks[b]:
                    v[b] += weight
                else:
                    v[b] -= weight

        fingerprint = 0
        for b in range(self.hashbits):
            if v[b] > 0:
                fingerprint |= masks[b]
        return fingerprint


def calculate_distance(right, left, hashbits=128):
    x = (right ^ left) & ((1 << hashbits) - 1)
    result = 0
    while x:
        result += 1
        x &= x - 1
    return result

def similarity(right_hash, left_hash, hashbits=128):
    return (1 - calculate_distance(right_hash, left_hash, hashbits) / float(hashbits))

if __name__ == "__main__":
    set_url_removed = set()
    #path_to_db = Path("../DEV")
    path_to_db = Path("/home/lopes/Datasets/IR/DEV")

    rf = open("duplicate.txt", "w+")
    for dir in path_to_db.iterdir():
        if not dir.is_dir():
            continue
        file_list = [f for f in dir.iterdir() if f.is_file()]
        sim_list = []
        url_list = []
        for file in file_list:
            with open(file, 'r', encoding="ascii", errors="ignore") as f:
                parsed_json = json.load(f)
                content = parsed_json['content']
                url = parsed_json['url']
                url_list.append(url)
                simhash = Simhash(content)
                sim_list.append((simhash.hash, simhash.size))

        for i in range(0, len(file_list) - 1):
            for j in range(i + 1, len(file_list)):
                sim = similarity(sim_list[i][0], sim_list[j][0])
                if sim >= 0.9:
                    rf.write("%s : %s\n" %(url_list[i], url_list[j]))

                    if sim_list[i][1] > sim_list[j][1]:
                        set_url_removed.add(url_list[j])
                    else:
                        set_url_removed.add(url_list[i])
                        break

    rf.close()
    with open("database/removed_urls.pkl", "wb") as fp:
        pickle.dump(set_url_removed, fp)
