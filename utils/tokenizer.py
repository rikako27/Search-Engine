from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
from bs4 import BeautifulSoup
from collections import defaultdict

class Tokenizer:
    def __init__(self, content: str, ngram):
        self.content = content
        self.importance = {'title': 6.0, 'h1': 5.0, 'h2': 4.0, 'h3': 3.0, 'b': 2.0}
        self.ngram = ngram
        self.length = 0

    def tokens(self, text: str) -> list:
        #split text into all alphanumeric tokens and perform porter stemming
        tokenizer = RegexpTokenizer('[0-9A-Za-z]+')
        # not allow digits that has length > 4
        token = [t for t in tokenizer.tokenize(text) if not (t.isdigit() and len(t) >= 5)]
        stemmer = PorterStemmer()
        terms_list = [stemmer.stem(t).lower() for t in token]
        if self.ngram > 1:
            terms_list = list(ngrams(terms_list, self.ngram))
        self.length = len(terms_list)
        return terms_list

    def calculate_tf(self, word_freq: dict):
        tf = defaultdict(float)
        total_words = sum(word_freq.values())
        for word, freq in word_freq.items():
            tf[word] = round(freq / total_words, 8)
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
            text.extract()

        all_texts = self.tokens(bs.get_text().lower())
        for text in all_texts:
            word_freq[text] += 1.0
        return self.calculate_tf(word_freq)
