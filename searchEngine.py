from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from bs4 import BeautifulSoup
from collections import defaultdict, Counter
import pickle

class SearchEngine:
    def __init__(self, query, top=5):
        self.query_tokens = self.tokens(query) #query should be tokenized and stemmed
        self.doc_length = 41128
        self.scores = defaultdict(float)
        self.doc = defaultdict(list)
        self.top = top

    def tokens(self, text: str) -> list:
        #split text into all alphanumeric tokens and perform porter stemming
        tokenizer = RegexpTokenizer('[0-9A-Za-z]+')

        # not allow digits that has length > 4
        token = [t for t in tokenizer.tokenize(text) if not (t.isdigit() and len(t) >= 5)]
        stemmer = PorterStemmer()
        return [stemmer.stem(t).lower() for t in token]

    def cosine_score(self, ngram):
        for t in self.query_tokens:
            weight_term_query = 1.0 / len(self.query_tokens)
            file = "database/" + t[0] + str(ngram)
            read_file = open(file, 'r')
            term_dict = read_file[t]

            for doc_id, tfidf in term_dict:
                self.scores[doc_id] += tfidf * weight_term_query

        with open("database/doc_id.txt", "r") as f:
            for line in f:
                line = line.rstrip('\n').split(', ')
                self.doc[int(line[0])] = [line[1], int(line[2])] #append url and doc_length
            for doc_id in self.scores:
                self.scores[doc_id] = self.scores[doc_id] / self.doc[doc_id][1]

        return dict(Counter(self.scores).most_common(self.top))

    def print_result(self):
        result_dict = self.cosine_score()
        index = 1
        for doc_id, tfidf in sorted(result_dict.items(), key=(lambda d: (-d[1], d[0]))):
            round_tfidf = round(tfidf + 0.005, 5)
            print("%d. url: %s  (tfidf = %f)\n" % (index, self.doc[doc_id], round_tfidf))
            index += 1

if __name__ == '__main__':
    while True:
        user_query = input("Enter query (Type q to stop program): ")
        if user_query in ('q', 'Q'):
            print("End of Program")
            break

        search_engine = SearchEngine(user_query)
        result_dict = search_engine.print_result()
