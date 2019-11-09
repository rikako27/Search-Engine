from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from bs4 import BeautifulSoup
from collections import defaultdict
import re

class Tokenizer:
    def __init__(self, content: str):
        self.content = content
        self.importance = {'title': 6.0, 'h1': 5.0, 'h2': 4.0, 'h3': 3.0, 'b': 2.0}
        self.data = {}

    def tokens(self, text: str) -> list:
        #split text into all alphanumeric tokens and perform porter stemming
        tokenizer = RegexpTokenizer('[0-9A-Za-z]+', flags=re.UNICODE)

        # not allow digits that has length > 4
        token = [t for t in tokenizer.tokenize(text) if not (t.isdigit() and len(t) >= 5)]
        stemmer = PorterStemmer()
        return [stemmer.stem(t).lower() for t in token]

    def calculate_tf(self, word_freq: dict):
        tf = defaultdict(float)
        total_words = sum(word_freq.values())
        for word, freq in word_freq.items():
            tf[word] = freq / total_words
        return tf

    def extract_texts(self):
        word_freq = defaultdict(float)
        bs = BeautifulSoup(self.content, 'lxml')

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
