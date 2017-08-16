#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

import json
import os
import sys
import jieba
import _uniout
import numpy as np
import csv
import cPickle as pickle
import argparse
import time

from client.cn_util import print_cn

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
from client.query_util import QueryUtils

reload(sys)
sys.setdefaultencoding("utf-8")


class Belief_MultiLabel_Clf:
    def __init__(self, data_path):
        self.data_path = data_path
        self.clf = None
        self._build()

    def _build_feature_extraction(self, mode, data_path):
        print('Build feature extraction...')
        corpus = list()
        with open(data_path, 'r') as f:
            reader = csv.reader(f, delimiter='#')
            for line in reader:
                b = line[1].encode('utf-8')
                tokens = self.cut(b)
                corpus.append(tokens)

        if mode == 'ngram':
            print_cn('Use {0}'.format(mode))
            bigram_vectorizer = CountVectorizer(
                ngram_range=(1, 2), min_df=0.0, max_df=1.0, analyzer='char', stop_words=[',', '?', '我', '我要'],
                binary=True)
            self.feature_extractor = bigram_vectorizer.fit(corpus)
        if mode == 'tfidf':
            print_cn('Use {0}'.format(mode))
            tfidf_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 2), max_df=1.0, min_df=1,
                                               sublinear_tf=True)
            self.feature_extractor = tfidf_vectorizer.fit(corpus)

    def cut(self, input_):
        input_ = QueryUtils.static_remove_cn_punct(input_)
        tokens = jieba.cut(input_, cut_all=True)
        seg = " ".join(tokens)
        tokens = _uniout.unescape(str(seg), 'utf8')
        return tokens

    def _build(self):
        self._build_feature_extraction('tfidf', self.data_path)
        mlb = MultiLabelBinarizer()
        embeddings = list()
        labels = list()

        with open(self.data_path, 'r') as f:
            reader = csv.reader(f, delimiter='#')
            for line in reader:
                key = line[0].encode('utf-8')
                input_ = line[1].encode('utf-8')
                intention_list = key.split(",")
                tokens = self.cut(input_)
                # embedding = self.feature_extractor.transform(tokens).toarray()
                embeddings.append(tokens)
                labels.append(intention_list)

        embeddings = self.feature_extractor.transform(embeddings).toarray()
        self.mlb = mlb.fit(labels)
        labels_ = self.mlb.transform(labels)
        return embeddings, labels_

    def train(self):
        print('prepare data...')
        embeddings, labels_ = self._build()
        print("Training classifier")

        begin = time.clock()

        self.clf = OneVsRestClassifier(GradientBoostingClassifier(max_depth=8, n_estimators=1000))
        self.clf.fit(embeddings, labels_)

        end = time.clock()

        print('Train done. time: {0} mins'.format((end - begin) / 60))
        self.test(self.data_path)

    def predict(self, input_):
        tokens = [self.cut(input_)]
        embeddings = np.reshape(
            self.feature_extractor.transform(tokens).toarray()[0], [1, -1])
        prediction = self.clf.predict(embeddings)
        prediction_index_first_sample = np.where(prediction[0] == 1)
        labels = self.mlb.inverse_transform(prediction)
        # print_cn(labels)
        probs = self.clf.predict_proba(embeddings)

        ## note that in prediction stage, n_samples == 1
        return labels[0], probs[0][prediction_index_first_sample]

    def test(self, test_path):
        correct = 0.0
        total = 0.0

        with open(test_path, 'r') as f:
            reader = csv.reader(f, delimiter='#')
            for line in reader:
                # print_cn(line)
                key = line[0].encode('utf-8')
                input_ = line[1].encode('utf-8')
                labels = key.split(",")

                prediction, proba = self.predict(input_)
                if set(prediction) == set(labels):
                    correct = correct + 1
                else:
                    print('{0}: {1}-->{2}'.format(input_, ' '.join(labels), ' '.join(list(prediction))))
                total = total + 1
        print('Accuracy:{0}'.format(correct / total))

    @staticmethod
    def get_instance(model_path):
        with open(model_path, "rb") as input_file:
            clf = pickle.load(input_file)
            return clf


def train(train_data_path, model_path):
    clf = Belief_MultiLabel_Clf(train_data_path)
    clf.train()
    with open(model_path, 'wb') as pickle_file:
        pickle.dump(clf, pickle_file, pickle.HIGHEST_PROTOCOL)


def test(test_data_path, model_path):
    with open(model_path, "rb") as input_file:
        clf = pickle.load(input_file)
        clf.test(test_data_path)


def main():
    model_path = '../../model/sc/belief_clf.pkl'
    train_data_path = '../../data/sc/train/sale_v2.txt'
    test_data_path = '../../data/sc/train/sale_v2.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', choices={'train', 'test'},
                        default='train', help='mode.if not specified,it is in test mode')

    args = parser.parse_args()

    if args.m == 'train':
        train(train_data_path, model_path)
    elif args.m == 'test':
        test(test_data_path, model_path)
    else:
        print('Unknow mode, exit.')


if __name__ == '__main__':
    # main()

    model_path = '../model/sc/belief_clf.pkl'
    clf = Belief_MultiLabel_Clf.load(model_path=model_path)
    inputs=["买点零食",'零食','麻辣小龙虾','我想买点实惠的面包','我要吃点实惠的牛排','买实惠的辣的单鞋','我要买衣服和鞋子','吃火锅','我过来买东西','我要买东西','购物']
    for p in inputs:
        labels, probs = clf.predict(input_=p)
        print_cn(labels, probs)
