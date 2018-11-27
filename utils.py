import string
from string import punctuation

from nltk.corpus import stopwords


def sent2labels(sent):
    return [label for token, label in sent]


def sent2tokens(sent, lower=False):
    tokens = []
    for s in sent:
            tokens.append(s[0].lower() if lower else s[0])

    return tokens


def pred2label(pred):
    return [label for token, label in pred]


def pre_processa(tokens):

    stop_words = set(stopwords.words('english'))

    stop_chars = {s for s in string.ascii_lowercase}
    stop_punct = {s for s in string.punctuation}
    stops = stop_words.union(stop_chars).union(stop_punct)

    with open('./data/stopwords_en.txt') as arq:
        stops = stops.union(set(arq.read().split('\n')))

    # lower case
    tokens = [t.lower() for t in tokens]

    # remove pontuação e stop words
    return [t for t in tokens if t not in punctuation and t not in stop_words]


def save_file_train(data_filename, data):

    with open(data_filename, 'w') as arq:
        for s in data:

            for k in s:
                arq.write('%s %s\n' % (k[0], k[1]))
            arq.write('\n')


def read_data(data_file_path):

    with open(data_file_path) as arq:

        sents = arq.read().strip().split('\n\n')
        data = []
        for sent in sents:
            sent = sent.strip().split('\n')
            sentence = []
            for tag in sent:
                sentence.append(tuple(tag.strip().split()))
            data.append(sentence)
        return data


if __name__ == '__main__':
    arq = read_data('./data/train.data')

    sent = sent2tokens(arq[1])
    print(pre_processa(sent))
