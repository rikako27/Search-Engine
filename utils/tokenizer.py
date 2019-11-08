from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from collections import defaultdict

class Tokenizer:
    def __init__(self, content: str):
        self.content = content
        self.ps = PorterStemmer()
        #Words in bold, in headings (h1, h2, h3), and in titles should be important
        self.importance = {'title': 5.0, 'h1': 3.0, 'h2': 2.0, 'h3': 1.5, 'b': 1.0}
        self.position_index = defaultdict(list)
        self.word_freq = defaultdict(float)
        self.tf = defaultdict(float)


    def tokens(self, text: str) -> list:
        #split text into all alphanumeric tokens and perform porter stemming
        tokenizer = RegexpTokenizer('\w+')
        stemmer = PorterStemmer()
        return [stemmer.stem(t) for t in tokenizer.tokenize(text)]

    def extract_texts(self):
        bs = BeautifulSoup(self.content, 'lxml')

        #delete <script> and <style> elements
        for script in bs(["script", "style"]):
            script.extract()

        #find header, bold, and title, and add weights
        for text in bs.find_all(['title', 'h1', 'h2', 'h3', 'b']):
            tag_name = text.name
            list_tokens = self.tokens(text.get_text().lower())
            for token in list_tokens:
                self.word_freq[token] += self.importance[tag_name]

        all_texts = self.tokens(bs.get_text().lower())
        for index, text in enumerate(all_texts, 1):
            self.position_index[text].append(index)
            self.word_freq[text] += 1.0

    def calculate_tf(self):
        total_words = sum(self.word_freq.values())
        for word, freq in self.word_freq.items():
            self.tf[word] = freq / total_words
        return self.tf

    def get_positional_index(self):
        return self.position_index
