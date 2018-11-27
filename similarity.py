import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import sent2tokens
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class Sent2Vec:

    def __init__(self, model_pre=None, sentences=None, *args, **kwargs):

        # Amarzena sentenças
        self.sentences = sentences if sentences else []

        if model_pre:
            self.model = model_pre
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
        self.vectors = self.model.wv

        # Calcula vectors para sentenças
        self._docs2vec()

    def _docs2vec(self):
        self.matrix = [self.sent2vec(sent) for sent in self.sentences]

        return np.array(self.matrix)

    def sent2vec(self, sentence):
        ''' '''
        vec = [self.vectors[s] for s in sentence if s in self.vectors]

        # Soma todos vector de uma sentença
        return np.sum(vec, axis=0)

    def distance(self, sentence_token):
        ''' Cosine '''
        return cosine_similarity([self.sent2vec(sentence_token)], self.matrix)


class Document2Vec:
    ''' Document2Vec'''

    max_epochs = 100
    vec_size = 100
    alpha = 0.025

    def __init__(self, documents=None, *args, **kwargs):

        self.tagged_data = [
            TaggedDocument(words=sent2tokens(_d, lower=True), tags=[str(i)])
            for i, _d in enumerate(documents)
        ]

        self.model = Doc2Vec(
            vector_size=self.vec_size, alpha=self.alpha, min_alpha=0.00025,
            min_count=1, dm=1, *args, **kwargs
        )

        self.model.build_vocab(self.tagged_data)

        self.model.train(
            self.tagged_data, total_examples=self.model.corpus_count,
            epochs=self.model.iter
        )

    def get_vector(self, sent):
        ''''''
        self.model.docvecs.infer_vector(sent)


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

    def eval(self, document, lim_informative):
        y = self.vectorizer.transform([document])

        sm_cosine = cosine_similarity(y, self.X)[0]

        return max(sm_cosine)
