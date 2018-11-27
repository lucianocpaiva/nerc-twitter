import logging
import numpy as np
import argparse

from utils import sent2tokens, sent2labels, read_data
from utils import pre_processa, save_file_train
from similarity import Tfidf
from stanford import StanfordNERTagger

_JAR = './lib/stanford-ner.jar'
_CLASSIFIER = './classifiers/classifier_1.stanford'

# Ative o treinamento quanto atigir esse número de novos dados
_DATA_ADD = 100


def val(classifier, test):
    ''' Valida um classificador '''

    X_test = [sent2tokens(xseq) for xseq in test]
    y_test = [sent2labels(xseq) for xseq in test]

    preds = classifier.predict(X_test)

    return classifier.bio_classification_report(y_test, preds)


def active_self_learning(data_inicial, test, stream_data, classifier, tfidf,
    lim_informative, lim_confidence):

    logging.debug('Limite_informativo: %f\tLimite confiança: %f',
        lim_informative, lim_confidence)

    n_round = 0
    count_sim = 0
    count_prob = 0
    count_data = 0

    # Número de partições do dados de consulta
    cut_stream = len(stream_data) // _DATA_ADD

    # Percorre os dados de consulta selecionando novos dados de treino
    for stream in np.array_split(np.array(stream_data), cut_stream):

        array_tokens = np.array([pre_processa(sent2tokens(s)) for s in stream])

        # Calcula similaridade de cada sentença com as da base de treinamento
        similarity = np.array([tfidf.eval(t, lim_informative) for t in array_tokens])

        # Seleciona todos que a similaridade seja inferior ao limite
        index_sim_min = similarity < lim_informative
        data_inicial += list(stream[index_sim_min])

        # Seleciona sentenças que a similaridade seja superior ao limite
        array_tokens = array_tokens[~index_sim_min]

        # Calcula confiança do classificador
        probs = np.array(classifier.probability_sent(array_tokens))

        # Seleciona todos que tenham confiança acima do limite
        index_probs_max = probs > lim_confidence
        array_tokens = array_tokens[index_probs_max]

        if len(array_tokens) != 0:
            preds = classifier.predict(array_tokens)

            data = []
            for tokens, taggers in zip(array_tokens, preds):
                data.append([(token, tag) for token, tag in zip(tokens, taggers)])

            data_inicial += data

        # Conta número de elementos adicionados por similaridade e confiança
        count_sim += np.sum(index_sim_min)
        count_prob += np.sum(index_probs_max)

        # conta número de elementos já selecionados
        count_data = np.sum(index_sim_min) + np.sum(index_probs_max)

        # Quando chegar a um número de novas sentenas treina classificador
        if count_data >= _DATA_ADD:
            count_data = 0
            n_round += 1

            logging.info(
                '\nRodada %d\t Tam conjunto treino: %d\
                \nBaixa similaridade: %d, Alta confiança: %d',
                n_round, len(data_inicial), int(count_sim), int(count_prob)
            )

            # Sava novos dados de treinamento
            save_file_train('./data/train_clean', data_inicial)

            # Treina tfidf
            X_train = [sent2tokens(xseq) for xseq in data_inicial]
            tfidf.train(documents=X_train)

            classifier.fit()

            # Avalia novo modelo
            logging.info('\n%s', val(classifier, test))

    return classifier


def help():
    ''' Trata argumentos inseridos pelo usuário, exibe texto de ajuda '''
    # Descrição do programa.
    parser = argparse.ArgumentParser()

    # Adicinando argumento.
    parser.add_argument('--log', '-l', action='store', dest='log_file',
        default=None, required=False, help='Arquivo de log')

    return parser.parse_args()


if __name__ == '__main__':

    arguments = help()

    logging.basicConfig(filename=arguments.log_file, level=logging.DEBUG,
        format='%(message)s')

    # Abrindo file prediction
    train = read_data('./data/train_clean')
    test = read_data('./data/dev_clean')
    stream_data = read_data('./data/test_clean')

    logging.info('Started\nTreino: %d\tTeste: %d\tConsulta: %d',
        len(train), len(test), len(stream_data))

    # Instância e treina classificador com dados iniciais
    classifier = StanfordNERTagger(_CLASSIFIER, _JAR, encoding='utf8')
    classifier.fit()

    # Avalia modelo e grava resultado em log
    logging.info('\n%s', val(classifier, test))

    # Calcula tf-idf dos dados iniciais de treinamento
    X_train = [sent2tokens(xseq) for xseq in train]
    tfidf = Tfidf()
    tfidf.train(documents=X_train)

    # Aplica abordagem active e self learning
    clr = active_self_learning(
        data_inicial=train,
        test=test,
        stream_data=stream_data,
        classifier=classifier,
        tfidf=tfidf,
        lim_informative=0,
        lim_confidence=0.5,
    )
