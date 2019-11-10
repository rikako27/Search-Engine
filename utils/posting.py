class Posting:
    def __init__(self, doc_id: int, tf_idf: float, next_node=None):
        self.doc_id = doc_id
        self.tf_idf = tf_idf
        self.next_node = next_node

    def get_docid(self):
        return self.doc_id
        
    def get_tf_idf(self):
        return self.get_tf_idf

    def get_next(self):
        return self.next_node

    def set_next(self, new_next):
        self.next_node = new_next
