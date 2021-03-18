from typing import List, Dict, Set
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import euclidean_distances

class Word2VecTrainer:
    # define training data
    sentences = []
    model = None
    tensor = None
    keyed_vectors = []
    word_list = []
    child_embedding_dict = {}
    ignore_list = []

    def __init__(self):
        self.reset()

    def reset(self):
        self.sentences = []
        self.model = None
        self.tensor = None
        self.keyed_vectors = []
        self.word_list = []
        self.child_embedding_dict = {}

    def set_ignore_list(self, ignore_list: List):
        self.ignore_list = ignore_list.copy()

    def add_text(self, text: str):
        print("add_text(): {}".format(text))
        raw = text.strip().split(" ")
        sentence = []
        for w in raw:
            if w not in self.ignore_list:
                sentence.append(w)

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

    def get_word_vectors(self, term: str, threshold: float) -> List:
        to_return = []
        term = term.lower()
        term_vec = []
        try:
            term_vec = self.keyed_vectors[term]
        except KeyError:
            return []

        for key in self.model.wv.vocab:
            try:
                if key != term:
                    vec = self.keyed_vectors[key]
                    dist = euclidean_distances([vec], [term_vec])
                    if dist[0][0] < threshold:
                        to_return.append(key)
            except KeyError:
                continue

        return to_return


    def get_text_vector(self, text: str, ignore: List = []) -> List:
        word_list = text.strip().split(" ")
        vec_list = []
        self.word_list = []
        for w in word_list:
            try:
                result = self.get_word_vector(w)
                vec_list.append(result)
                self.word_list.append(w)
            except KeyError:
                continue

        if len(vec_list) == 0:
            return []

        mat = np.array(vec_list)
        vec = mat.mean(0)
        return vec.tolist()

    def get_words(self) -> str:
        return " ".join(self.word_list)

    def clear_child_embeddings(self):
        self.child_embedding_dict = {}

    def add_child_embedding(self, name:str, word_list:list, carray:List):
        vec_list = []
        for w in word_list:
            try:
                result = self.get_word_vector(w)
                vec_list.append(result)
            except KeyError:
                continue
        if len(vec_list) > 1:
            dict = {"vec_list":vec_list, "carray":carray}
            self.child_embedding_dict[name] = dict
            print("added {} child embeddings for {}".format(len(vec_list), name))


    def plot_child_embeddings(self, show: bool = True, show_context:bool = False, size=20):
        print ('drawing {} embeddings'.format(len(self.child_embedding_dict)))
        nt:np.array = np.array(self.tensor)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for key in self.child_embedding_dict:
            print("drawing embedding {}".format(key))
            dict = self.child_embedding_dict[key]
            vec_list = np.array(dict["vec_list"])
            carray = dict["carray"]
            x = vec_list[:,0]
            y = vec_list[:,1]
            if nt.shape[1] > 2:
                z = vec_list[:,2]
                ax.scatter(x, y, z, c=carray, label=key, s=size)
            else:
                ax.scatter(x, y, c=carray, label=key, s=size)
        if show_context:
            if nt.shape[1] > 2:
                ax.scatter(xs=self.tensor[:, 0], ys=self.tensor[:, 1], zs=self.tensor[:, 2], c=[0.3, 0.3, 0.3, 0.3], s=size)
            else:
                ax.scatter(xs=self.tensor[:, 0], ys=self.tensor[:, 1], c=[0.3, 0.3, 0.3, 0.3], s=size)
        if show:
            plt.title("Child Embeddings")
            plt.legend(loc='best')
            plt.show()

    def plot(self, show: bool = True, carray:List = None, size=20):
        print ('drawing {} points'.format(len(self.tensor)))
        nt:np.array = np.array(self.tensor)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if carray:
            if nt.shape[1] > 2:
                ax.scatter(xs=nt[:, 0], ys=nt[:, 1], zs=nt[:, 2], c = carray, s=size)
            else:
                ax.scatter(xs=nt[:, 0], ys=nt[:, 1], zs=nt[:, 2], c = carray, s=size)
        else:
            if nt.shape[1] > 2:
                ax.scatter(xs=nt[:, 0], ys=nt[:, 1], zs=nt[:, 2], s=size)
            else:
                ax.scatter(xs=nt[:, 0], ys=nt[:, 1], s=size)
        if show:
            plt.title("Master Embeddings")
            plt.show()

if __name__ == '__main__':
    filename = "D:/Development/Sandboxes/GPT-2_agents/data/chess/narrative_eval.txt"
    savedfile = "D:/Development/Sandboxes/GPT-2_agents/data/chess/sequences1.bin"

    row_list = []
    cols = ['king', 'queen', 'rook', 'bishop', 'knight', 'pawn']
    for i in range(1,9):
        row = []
        for c in cols:
            row.append("{}{}".format(c, i))
        row_list.append(row)

    col_list = []
    for c in cols:
        col = []
        for i in range(1, 9):
            col.append("{}{}".format(c, i))
        col_list.append(col)

    wv = Word2VecTrainer()
    with open(filename) as f:
        text = f.readlines()
        for t in text:
            wv.add_text(t.strip())
        wv.train(size=2)
        for i in range(len(col_list)):
            col = col_list[i]
            carray = [random.random(), random.random(), random.random()]
            name = "col {}".format(cols[i])
            wv.add_child_embedding(name, col, carray)
        wv.plot_child_embeddings(size=50)
        wv.save("../data/chess/sequences1")


    """
    v1 = [[0, 0]]
    v2 = [[1, 1]]
    dist = euclidean_distances(v1, v2)

    print("dist = {}".format(dist[0][0]))

    v1 = np.random.rand(10, 3)
    v2 = np.random.rand(10, 3)+1

    print(v1)

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    ax1.scatter(v1[:,0], v1[:,1], v1[:,2], c='b', marker="s", label='first')
    ax1.scatter(v2[:,0], v2[:,1], v2[:,2], c='r', marker="o", label='second')
    plt.legend(loc='best')
    plt.show()
    """