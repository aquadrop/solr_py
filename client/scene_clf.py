#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import csv
import jieba
import json
import _uniout
import cn_util
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from sklearn import metrics
import cPickle as pickle
import argparse


reload(sys)
sys.setdefaultencoding('utf-8')


class SceneClassifier(object):

    def __init__(self):
        self.kernel = None
        self.embeddings = list()
        self.labels = list()
        self.named_labels = ['business', 'qa', 'interaction']

    def _bulid_ngram(self, busi_path, qa_path, hudong_path):
        print 'build ngramer...'

        corpus = list()

        for path in [busi_path, qa_path, hudong_path]:
            with open(path, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for line in reader:
                    # print path, cn_util.print_cn(line)
                    line = str(line).strip()
                    b = line.encode('utf-8')
                    # print(b)
                    tokens = self.cut(b)

                    corpus.append(tokens)

        bigram_vectorizer = CountVectorizer(
            ngram_range=(1, 2), min_df=0.0, max_df=1.0, analyzer='char',
            stop_words=[',', '?', '我', '我要'], binary=True)

        self.bigramer = bigram_vectorizer.fit(corpus)

    def cut(self, input_):
        seg = " ".join(jieba.cut(input_, cut_all=False))
        tokens = _uniout.unescape(str(seg), 'utf8')
        return tokens

    def _prepare_data(self, busi_path, qa_path, hudong_path):
        print 'prepare data...'

        embeddings = list()
        labels = list()

        for path in [busi_path, qa_path, hudong_path]:
            with open(path, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for line in reader:
                    line = str(line).strip()
                    b = line.encode('utf-8')
                    # print(b)
                    tokens = [self.cut(b)]
                    embedding = self.bigramer.transform(tokens).toarray()
                    embeddings.append(embedding)

                    if path == busi_path:
                        label = 0
                    elif path == qa_path:
                        label = 1
                    else:
                        label = 2

                    labels.append(label)

        self.embeddings = np.array(embeddings)
        self.embeddings = np.squeeze(self.embeddings)

        self.labels = np.array(labels)

        self.embeddings, self.labels = shuffle(
            self.embeddings, self.labels, random_state=0)

    def build(self, busi_path, qa_path, hudong_path):
        self._bulid_ngram(busi_path, qa_path, hudong_path)
        self._prepare_data(busi_path, qa_path, hudong_path)

    def train(self, pkl):
        print 'train classifier...'

        self.kernel = GradientBoostingClassifier(max_depth=5, n_estimators=200)
        self.kernel.fit(self.embeddings, self.labels)

        pickle.dump(self, open(pkl, 'wb'))

        print 'train done and saved.'
        self.metrics_()

    def metrics_(self):
        pre = self.kernel.predict(self.embeddings)

        print metrics.confusion_matrix(self.labels, pre)

        # precision_score = metrics.precision_score(self.labels, pre)
        # recall_score = metrics.recall_score(self.labels, pre)
        # f1_score = metrics.f1_score(self.labels, pre)

        # print 'precision_score:{0}, recall_score:{1},
        # f1_score:{2}'.format(precision_score, recall_score, f1_score)

    def find_wrong(self):
        pass

    def predict(self, question):
        # clf = pickle.load(open('../model/bqclf.pkl', 'r'))
        embedding = self.bigramer.transform(
            [self.cut(question)]).toarray()
        embedding = np.squeeze(embedding)
        embedding = np.reshape(embedding, [1, -1])
        label = self.kernel.predict(embedding)[0]
        probs = self.kernel.predict_proba(embedding)
        # print probs
        # print prob
        return self.named_labels[label], probs

    def interface(self, q):
        label, probs = self.predict(q)
        probs_dict = {}
        for i in xrange(len(probs)):
            probs_dict[self.named_labels[i]] = probs[i]
        return label, probs_dict

    @staticmethod
    def get_instance(path):
        print('loading model file...')
        return pickle.load(open(path, 'r'))


def train():
    clf = SceneClassifier()
    clf.build('../data/scene/business_q.txt',
                       '../data/scene/common_qa_q.txt', '../data/scene/interactive-g.txt')
    clf.train('../model/scene/sceneclf.pkl')


def online_validation():
    clf = SceneClassifier.get_instance('../model/scene/sceneclf.pkl')
    print('loaded model file...')
    try:
        while True:
            question = raw_input('input something...\n')
            print clf.predict(question)[0]
            print 'prediction:{0}'.format(clf.predict(question))
    except KeyboardInterrupt:
        print('interaction interrupted')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices={'train', 'validation'},
                        default='validation', help='mode.if not specified,it is in prediction mode')
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'validation':
        online_validation()


if __name__ == '__main__':
    main()
