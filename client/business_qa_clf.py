#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import csv
import jieba
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


class BqClassifier(object):

    def __init__(self, path1, path2, path3):

        self.busi_path = path1
        self.qa_path = path2
        self.hudong_path = path3
        self.embeddings = list()
        self.labels = list()

    def bulid_ngram(self):
        print 'build ngramer...'

        corpus = list()

        for path in [self.busi_path, self.qa_path, self.hudong_path]:
            with open(path, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for line in reader:
                    # print path, cn_util.print_cn(line)
                    b = line[1].encode('utf-8')
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

    def prepare_data(self):
        print 'prepare data...'

        embeddings = list()
        labels = list()

        for path in [self.busi_path, self.qa_path, self.hudong_path]:
            with open(path, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for line in reader:
                    b = line[1].encode('utf-8')
                    # print(b)
                    tokens = [self.cut(b)]
                    embedding = self.bigramer.transform(tokens).toarray()
                    embeddings.append(embedding)

                    if path == self.busi_path:
                        label = 0
                    elif path == self.qa_path:
                        label = 1
                    else:
                        label = 2

                    labels.append(label)

        self.embeddings = np.array(embeddings)
        self.embeddings = np.squeeze(self.embeddings)

        self.labels = np.array(labels)

        self.embeddings, self.labels = shuffle(
            self.embeddings, self.labels, random_state=0)

    def build(self):
        self.bulid_ngram()
        self.prepare_data()

    def train(self):
        print 'train classifier...'

        self.clf = GradientBoostingClassifier(max_depth=5, n_estimators=200)
        self.clf.fit(self.embeddings, self.labels)

        pickle.dump(self.clf, open('../model/bqclf.pkl', 'wb'))

        print 'train done and saved.'
        self.metrics_()

    def metrics_(self):
        pre = self.clf.predict(self.embeddings)

        print metrics.confusion_matrix(self.labels, pre)

        # precision_score = metrics.precision_score(self.labels, pre)
        # recall_score = metrics.recall_score(self.labels, pre)
        # f1_score = metrics.f1_score(self.labels, pre)

        # print 'precision_score:{0}, recall_score:{1},
        # f1_score:{2}'.format(precision_score, recall_score, f1_score)

    def find_wrong(self):
        pass

    def predict(self):
        self.bulid_ngram()
        clf = pickle.load(open('../model/bqclf.pkl', 'r'))
        try:
            while True:
                question = raw_input('input something...\n')
                embedding = self.bigramer.transform(
                    [self.cut(question)]).toarray()
                embedding = np.squeeze(embedding)
                print 'prediction:{0}'.format(clf.predict(embedding))
        except KeyboardInterrupt:
            print('interaction interrupted')


def train():
    clf = BqClassifier('../data/train_pruned_fixed.txt',
                       '../data/common_qa.txt', '../data/hudong.txt')
    clf.build()
    clf.train()


def prediction():
    clf = BqClassifier('../data/train_pruned_fixed.txt',
                       '../data/common_qa.txt', '../data/hudong.txt')
    clf.predict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices={'train', 'prediction'},
                        default='prediction', help='mode.if not specified,it is in prediction mode')
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'prediction':
        prediction()


if __name__ == '__main__':
    main()
