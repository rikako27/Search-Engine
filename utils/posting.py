class Posting:
    def __init__(self, doc_id: int, token_id: int, tf: float):
        self.doc_id = doc_id
        self.token_id = token_id
        self.tf = tf
        self.tf_idf = 0.0

    def calculate_tfidf(self, num_files, num_files_with_token):
        idf = log(1.0 * num_files / num_files_with_token)
        tf_idf = idf * self.tf
        self.tf_idf = round(tf_idf + 0.005, 2)
