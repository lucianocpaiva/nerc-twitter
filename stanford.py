# -*- coding: utf-8 -*-
# Natural Language Toolkit: Interface to the Stanford Part-of-speech and Named-Entity Taggers
#
# Copyright (C) 2001-2019 NLTK Project
# Author: Nitin Madnani <nmadnani@ets.org>
#         Rami Al-Rfou' <ralrfou@cs.stonybrook.edu>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT
# Obs.: Modulo alterado do nltk

"""
A module for interfacing with the Stanford taggers.
Tagger models need to be downloaded from https://nlp.stanford.edu/software
and the STANFORD_MODELS environment variable set (a colon-separated
list of paths).
For more details see the documentation for StanfordPOSTagger and
StanfordNERTagger.
"""

import os
import tempfile
from subprocess import PIPE
from six import text_type

from nltk.internals import config_java, java, _java_options
from nltk.tag.stanford import StanfordTagger
from sklearn.preprocessing import LabelBinarizer
from itertools import chain
from sklearn.metrics import classification_report


class StanfordNERTagger(StanfordTagger):
    """
        # TODO: documentar
    """

    _JAR = 'stanford-ner.jar'
    _FORMAT = 'tsv'

    def __init__(self, *args, **kwargs):
        super(StanfordNERTagger, self).__init__(*args, **kwargs)

    @property
    def _cmd(self):
        pass

    def probability(self, sentence):

        # Parser
        def parse_output(text, sentences=None):

            tag_prob = []

            # Output the tagged sentences
            for line in text.strip().split('\n'):
                line = line.split('\t')[2:]

                probs = [prob.split('=') for prob in line]
                t = [(p[0], float(p[1])) for p in probs]
                sorted(t, key=lambda x: float(x[1]), reverse=True)
                tag_prob.append(t[0])

            return tag_prob

        sentence = [(s, 'I') for s in sentence]
        _input = '\n'.join((' '.join(x) for x in sentence))
        _input_file_path = self._create_tmp_file(_input)

        cmd = [
            'edu.stanford.nlp.ie.crf.CRFClassifier',
            '-loadClassifier',
            self._stanford_model,
            '-testFile',
            _input_file_path,
            '-printprobs'
        ]
        return self.run(_input_file_path, cmd, parse_output)

    def _create_tmp_file(self, _input):
        encoding = self._encoding

        # Create a temporary input file
        _input_fh, _input_file_path = tempfile.mkstemp(text=True)

        # Write the actual sentences to the temporary input file
        _input_fh = os.fdopen(_input_fh, 'wb')

        if isinstance(_input, text_type) and encoding:
            _input = _input.encode(encoding)
        _input_fh.write(_input)
        _input_fh.close()

        return _input_file_path

    def probability_sent(self, sentences):

        # Parser
        def parse_output(text, sentences=None):
            probs = []
            for prob in text.split('<sentence')[1:]:
                probs.append(float(prob.split('prob=')[1].split('>')[0]))

            return probs

        _input = ''
        for sentence in sentences:
            _input += '\n\n'
            _input += '\n'.join([s + '\tI' for s in sentence])

        _input_file_path = self._create_tmp_file(_input)

        cmd = [
            'edu.stanford.nlp.ie.crf.CRFClassifier',
            '-loadClassifier',
            self._stanford_model,
            '-testFile',
            _input_file_path,
            '-kbest', '1'
        ]
        return self.run(_input_file_path, cmd, parse_output)

    def predict(self, sentences):

        # Parser
        def parse_output(text, sentences=None):
            if self._FORMAT == 'tsv':

                # Joint together to a big list
                tagged_sentences = []
                for tagged_sentence in text.strip().split('\n\n'):
                    tagged_sentences.append(
                        [sent.split('\t')[2] for sent in tagged_sentence.split('\n')])

                return tagged_sentences

            raise NotImplementedError

        _input = '\tI\n\n'.join(('\tI\n'.join(x) for x in sentences))
        _input += '\tI'

        _input_file_path = self._create_tmp_file(_input)

        cmd = [
            'edu.stanford.nlp.ie.crf.CRFClassifier',
            '-loadClassifier',
            self._stanford_model,
            '-testFile',
            _input_file_path,
            '-outputFormat',
            self._FORMAT,
        ]

        return self.run(_input_file_path, cmd, parse_output)

    def fit(self):
        cmd = [
            'edu.stanford.nlp.ie.crf.CRFClassifier',
            '-prop',
            './prop.txt',
        ]

        self.run(None, cmd, None)

    def run(self, _input_file_, _cmd, parse_output):

        encoding = self._encoding
        default_options = ' '.join(_java_options)
        config_java(options=self.java_options, verbose=False)

        cmd = list(_cmd)
        cmd.extend(['-encoding', encoding])

        # Run the tagger and get the output
        stanpos_output, _stderr = java(
            cmd, classpath=self._stanford_jar, stdout=PIPE, stderr=PIPE
        )

        stanpos_output = stanpos_output.decode(encoding)

        # Delete the temporary file
        if _input_file_:
            os.unlink(_input_file_)

        # Return java configurations to their default values
        config_java(options=default_options, verbose=False)

        return parse_output(text=stanpos_output) if parse_output else None

    def bio_classification_report(self, y_true, y_pred):
        """
        Classification report for a list of BIO-encoded sequences.
        It computes token-level metrics and discards "O" labels.
        """

        lb = LabelBinarizer()
        y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
        y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

        tagset = set(lb.classes_) - {'O'}
        tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
        class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

        return classification_report(
            y_true_combined,
            y_pred_combined,
            labels=[class_indices[cls] for cls in tagset],
            target_names=tagset,
        )
