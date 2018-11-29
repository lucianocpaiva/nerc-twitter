import numpy as np
from gensim.models import Word2Vec
import gensim
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import sent2tokens, pre_processa, pos_tag, read_data
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from scipy.sparse import csc_matrix


class Sent2Vec:

    def __init__(self, model_pre=None, sentences=None, *args, **kwargs):

        # Amarzena sentenças
        self.sentences = sentences if sentences else []

        if model_pre:
            self.model = gensim.models.KeyedVectors.load_word2vec_format(model_pre, binary=True)
            self.vectors = self.model.wv
            self._docs2vec()
        else:
            self.model = Word2Vec(sentences, *args, **kwargs)

    def train(self, *args, **kwargs):

        # Treina modelos
        self.model.train(self.sentences, *args, **kwargs)

        # Armazena array de vector
        self.vectors = self.model.wv

        # Calcula vectors para sentenças
        self._docs2vec()

    def update(self, sentences):

        # Incrementa o número de sentenças
        self.sentences = np.append(self.sentences, sentences, axis=0)

        # Update the model with new data.
        self.model.build_vocab(sentences, update=True)

        # Train model
        self.model.train(
            sentences, total_examples=self.model.corpus_count,
            epochs=self.model.iter
        )
        import pdb;pdb.set_trace()
        self.vectors = self.model.wv

        # Calcula vectors para sentenças
        self._docs2vec()

    def _docs2vec(self):
        self.matrix = [self.sent2vec(sent) for sent in self.sentences]
        self.matrix = [m for m in self.matrix if len(m)  > 10]
        return csc_matrix(self.matrix)

    def sent2vec(self, sentence):
        ''' '''
        vec = [self.vectors[s] for s in sentence if s in self.vectors]

        # Soma todos vector de uma sentença
        return np.sum(vec, axis=0) if len(vec) > 0 else []

    def eval(self, sentence_token):
        ''' Cosine '''

        tokens = self.sent2vec(sentence_token)
        if len(tokens) == 0:
            return 1
        sm_cosine = cosine_similarity([tokens], self.matrix)[0]
        return max(sm_cosine)


class Tfidf:

    def __init__(self, *args, **kargs):

        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            tokenizer=lambda sent: [t.lower() for t in sent],
            lowercase=False,
            stop_words='english',
            *args, **kargs
        )

    def train(self, documents):
        self.X = self.vectorizer.fit_transform(documents)

    def eval(self, document):
        y = self.vectorizer.transform([document])
        sm_cosine = cosine_similarity(y, self.X)[0]

        return max(sm_cosine)


if __name__ == '__main__':

    # train = read_data('./data/train_clean')
    # X_train = [sent2tokens(xseq) for xseq in train]

    # model = './embeddings/GoogleNews-vectors-negative300.bin.gz'

    # m = Sent2Vec(
    #     model_pre=model, sentences=X_train
    # )
    
    # sent = 'I odie coding Microsoft'.split()

    # distance = m.eval(pos_tag(sent))
    
    s1 = 'The car is driven on the road'.split() 
    s2 = 'The truck is driven on the highway'.split()

    m = Tfidf()
    m.train([s1, s2])

    print(m.eval(s1))