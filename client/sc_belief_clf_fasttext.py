#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import traceback
import requests
import json
import os
import sys
import argparse
import time

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

import jieba
import _uniout
import numpy as np
import csv
import cPickle as pickle

from cn_util import print_cn
from query_util import QueryUtils

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)


reload(sys)
sys.setdefaultencoding("utf-8")


class Multilabel_Clf:
    def __init__(self, mode):
        self.mode = mode
        self.weighted = True
        self.clf = None
        self.fasttext_url_weighted = "http://localhost:11425/fasttext/s2v?q={0}&w={1}"
        self.fasttext_url = "http://localhost:11425/fasttext/s2v?q={0}"

        self.stop_words = [',', '?', u'我', u'我要', u'我来', u'我想要']

        # self._build()

    def _build_feature_extraction(self, data_path):
        print('Build feature extraction...')
        corpus = list()
        with open(data_path, 'r') as f:
            reader = csv.reader(f, delimiter='#')
            for line in reader:
                b = line[1].decode('utf-8')
                b = QueryUtils.static_remove_stop_words(b)
                tokens = QueryUtils.static_jieba_cut(b)
                corpus.append(tokens)

        if self.mode == 'ngram':
            print_cn('Use {0}'.format(self.mode))
            bigram_vectorizer = CountVectorizer(
                ngram_range=(1, 2), min_df=0.0, max_df=1.0, analyzer='char', stop_words=[',', '?', '我', '我要'],
                binary=True)
            self.feature_extractor = bigram_vectorizer.fit(corpus)
        if self.mode == 'tfidf':
            print_cn('Use {0}'.format(self.mode))
            tfidf_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 2), max_df=1.0, min_df=1,
                                               sublinear_tf=True)
            self.feature_extractor = tfidf_vectorizer.fit(corpus)
        if self.mode == 'fasttext':
            pass

    # def cut(self, input_):
    #     input_ = QueryUtils.static_remove_cn_punct(input_)
    #     tokens = jieba.cut(input_, cut_all=True)
    #     seg = " ".join(tokens)
    #     tokens = _uniout.unescape(str(seg), 'utf8')
    #     return tokens

    def _fasttext_vector(self, tokens):
        if not self.weighted:
            url = self.fasttext_url.format(','.join(tokens))
        else:
            try:
                idf_url = "http://10.89.100.14:3032/s/{0}".format("%7C".join(tokens))
                idf_r = requests.get(url=idf_url)
                weights = []
                returned_json = idf_r.json()
                max_weight = 1
                for key, value in returned_json.iteritems():
                    if value > max_weight:
                        max_weight = value
                for token in tokens:
                    if token not in returned_json:
                        weights.append(str(max_weight))
                    else:
                        weights.append(str(returned_json[token]))

                url = self.fasttext_url_weighted.format(','.join(tokens), ','.join(weights))
            except:
                traceback.print_exc()
                url = self.fasttext_url.format(','.join(tokens))
        r = requests.get(url=url)
        vector = r.json()['vector']
        return vector

    def remove_stop_words(self, q):
        for stop in self.stop_words:
            q.replace(stop, '')
        return q

    def _build(self, data_path):
        self._build_feature_extraction(data_path)
        mlb = MultiLabelBinarizer()
        embeddings = list()
        labels = list()
        if self.mode == 'fasttext':
            with open(data_path, 'r') as f:
                reader = csv.reader(f, delimiter='#')
                for line in reader:
                    key = line[0].decode('utf-8')
                    input_ = line[1].decode('utf-8')
                    intention_list = key.split(",")
                    tokens = QueryUtils.static_jieba_cut(input_)
                    # embedding = self.feature_extractor.transform(tokens).toarray()
                    vector = self._fasttext_vector(tokens)
                    if not vector:
                        continue
                    embedding = vector
                    embeddings.append(embedding)
                    labels.append(intention_list)

            # embeddings = self.feature_extractor.transform(embeddings).toarray()
            self.mlb = mlb.fit(labels)
            labels_ = self.mlb.transform(labels)
            return embeddings, labels_
        else:
            with open(data_path, 'r') as f:
                reader = csv.reader(f, delimiter='#')
                for line in reader:
                    key = line[0].encode('utf-8')
                    input_ = line[1].encode('utf-8')
                    intention_list = key.split(",")
                    tokens = QueryUtils.static_jieba_cut(input_)
                    # embedding = self.feature_extractor.transform(tokens).toarray()
                    embeddings.append(tokens)
                    labels.append(intention_list)

            embeddings = self.feature_extractor.transform(embeddings).toarray()
            self.mlb = mlb.fit(labels)
            labels_ = self.mlb.transform(labels)
            return embeddings, labels_

    def train(self, data_path):
        print('prepare data...')
        embeddings, labels_ = self._build(data_path)
        print("Training classifier")

        begin = time.clock()

        self.clf = OneVsRestClassifier(GradientBoostingClassifier(max_depth=5, n_estimators=2000))
        self.clf.fit(embeddings, labels_)

        end = time.clock()

        print('Train done. time: {0} mins'.format((end - begin) / 60))
        self.test(data_path)

    def predict(self, input_):
        input_ = QueryUtils.static_remove_stop_words(input_)
        tokens = QueryUtils.static_jieba_cut(input_)
        try:
            if self.mode == 'fasttext':
                embedding = self._fasttext_vector(tokens)
                embeddings = np.array([embedding])
            else:
                embeddings = np.reshape(
                    self.feature_extractor.transform(tokens).toarray()[0], [1, -1])
        except:
            embeddings = np.reshape(
                self.feature_extractor.transform(tokens).toarray()[0], [1, -1])
        prediction = self.clf.predict(embeddings)
        prediction_index_first_sample = np.where(prediction[0] == 1)
        labels = self.mlb.inverse_transform(prediction)
        # print_cn(labels)
        probs = self.clf.predict_proba(embeddings)

        # note that in prediction stage, n_samples == 1
        return labels[0], probs[0][prediction_index_first_sample]

    def test(self, test_path):
        correct = 0.0
        total = 0.0

        with open(test_path, 'r') as f:
            reader = csv.reader(f, delimiter='#')
            for line in reader:
                # print_cn(line)
                key = line[0].decode('utf-8')
                input_ = line[1].decode('utf-8')
                labels = key.split(",")

                prediction, proba = self.predict(input_)
                if set(prediction) == set(labels):
                    correct = correct + 1
                else:
                    print('{0}: {1}-->{2}'.format(input_, ' '.join(labels), ' '.join(list(prediction))))
                total = total + 1
        print('Accuracy:{0}'.format(correct / total))

    @staticmethod
    def load(model_path):
        with open(model_path, "rb") as input_file:
            clf = pickle.load(input_file)
            return clf


def train(train_data_path, model_path, mode):
    clf = Multilabel_Clf(mode=mode)
    clf.train(train_data_path)
    with open(model_path, 'wb') as pickle_file:
        pickle.dump(clf, pickle_file, pickle.HIGHEST_PROTOCOL)


def test(test_data_path, model_path):
    with open(model_path, "rb") as input_file:
        clf = pickle.load(input_file)
        clf.test(test_data_path)


def main():
    mode = 'fasttext'
    model_path = '../model/sc/belief_clf_fasttext.pkl'
    train_data_path = '../data/sc/train/sale_v2.txt'
    test_data_path = '../data/sc/train/sale_v2.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', choices={'train', 'test'},
                        default='train', help='mode.if not specified,it is in test mode')

    args = parser.parse_args()

    if args.m == 'train':
        train(train_data_path, model_path, mode)
    elif args.m == 'test':
        test(test_data_path, model_path)
    else:
        print('Unknow mode, exit.')


if __name__ == '__main__':
    main()

    model_path = '../model/sc/belief_clf_fasttext.pkl'
    clf = Multilabel_Clf.load(model_path=model_path)
    inputs = ["我想买点糖果"]
    for p in inputs:
        labels, probs = clf.predict(input_=p)
        print('{0}:-->{1}'.format(p, ' '.join(labels)))
