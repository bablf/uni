from typing import List, Dict, Set
import numpy as np
import re
from gensim.models import Word2Vec, KeyedVectors
import gpt2agents.utils.MySqlInterface as MSI

class W2VEmbeddings:
    msi:MSI.MySqlInterface
    sentences:List
    model:Word2Vec
    tensor = None
    keyed_vectors:KeyedVectors
    word_list:List
    child_embedding_dict:Dict
    ignore_list:List
    ignore_partial_list:List

    def __init__(self, msi:MSI):
        self.msi = msi
        self.reset()

    def reset(self):
        print("W2VEmbeddings.reset")
        self.sentences = []
        self.model = None
        self.tensor = None
        self.keyed_vectors = None
        self.word_list = []
        self.ignore_list = []
        self.ignore_partial_list = []
        self.child_embedding_dict = {}

    def set_ignore_list(self, ignore_list: List):
        self.ignore_list = ignore_list.copy()

    def set_ignore_partial_list(self, l:List):
        self.ignore_partial_list = l.copy()

    def add_text(self, text: str, tolower:bool = False):
        # print("add_text(): {}".format(text))
        raw = text.strip().split(" ")
        sentence = []
        w:str
        for w in raw:
            if w not in self.ignore_list:
                present = False
                for partial in self.ignore_partial_list:
                    if partial in w:
                        present = True
                        break
                if not present:
                    if tolower:
                        w = w.lower()
                    sentence.append(w)
        # print(sentence)

        self.sentences.append(sentence)

    def train(self, workers:int = 4, size:int = 3):
        self.model = Word2Vec(self.sentences, workers=workers, size=size)
        self.tensor = self.model[self.model.wv.vocab]
        self.keyed_vectors = self.model.wv
        print("train: tensor length is {}".format(len(self.tensor)))

    def save(self, filename:str):
        self.model.save("{}.bin".format(filename))

    def load(self, filename:str):
        self.model = Word2Vec.load(filename)
        print ('loaded')
        self.tensor = self.model[self.model.wv.vocab]
        self.keyed_vectors = KeyedVectors.load(filename)

    def get_word_vector(self, key:str):
        return self.keyed_vectors[key]

    def create_embedding_from_query(self, sql:str, key_list:List, dimension:int = 3, strip_puncturation:bool = True) -> KeyedVectors:
        # print(sql)
        punct_regex = re.compile(r"[^\w\s]")
        result = self.msi.read_data(sql)
        for r in result:
            for k in key_list:
                word = r[k]
                if strip_puncturation:
                    word = punct_regex.sub("", word)
                self.add_text(word, tolower=True)
                #print(r[k])
        self.train(size=dimension)
        print("create_embedding_from_query(): trained")
        return self.keyed_vectors

def main():
    # user_name: str, user_password: str, db_name: str, enable_writes: bool = True
    key_list = ['content']

    msi = MSI.MySqlInterface(user_name="root", user_password="postgres", db_name="gpt_experiments", enable_writes=False)
    we = W2VEmbeddings(msi)
    we.set_ignore_partial_list(['http://', 'https://'])
    we.create_embedding_from_query("select content from table_output where experiment_id = 1 and tag = 'tweet';", key_list)
    msi.close()

if __name__ == "__main__":
    main()