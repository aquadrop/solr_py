#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import jieba
import json
import os

import _uniout
from cn_util import print_cn
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
import cPickle as pickle

import argparse

reload(sys)
sys.setdefaultencoding('utf-8')

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)


class SceneClassifier(object):
    def __init__(self):
        self.kernel = None
        self.named_labels = ['dialogue', 'greeting', 'qa']

    def _build_feature_extractor(self, mode, files):
        print('Build feature extraction...')
        corpus = list()

        for path in files:
            with open(path, 'r') as f:
                for line in f:
                    line = json.loads(line.strip().decode('utf-8'))
                    question = line['question']
                    tokens = self.cut(question)
                    corpus.append(tokens)

        if mode == 'ngram':
            bigram_vectorizer = CountVectorizer(
                ngram_range=(1, 2), min_df=0.0, max_df=1.0, analyzer='char', stop_words=[',', '?', '我', '我要'],
                binary=True)
            self.feature_extractor = bigram_vectorizer.fit(corpus)
        if mode == 'tfidf':
            print_cn('use {0}'.format(mode))
            tfidf_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 2), max_df=1.0, min_df=1,
                                               sublinear_tf=True)
            self.feature_extractor = tfidf_vectorizer.fit(corpus)

    def cut(self, input_):
        seg = " ".join(jieba.cut(input_, cut_all=False))
        tokens = _uniout.unescape(str(seg), 'utf8')
        return tokens

    # def _prepare_data(self, files):
    #     print ('prepare data...')
    #
    #     embeddings = list()
    #     queries = list()
    #     labels = list()
    #
    #     for index in xrange(len(files)):
    #         path = files[index]
    #         with open(path, 'r') as f:
    #             for line in f:
    #                 try:
    #                     line = json.loads(line.strip().decode('utf-8'))
    #                     question = line['question']
    #                     if question:
    #                         tokens = [self.cut(question)]
    #                         embedding = self.feature_extractor.transform(
    #                             tokens).toarray()
    #                         embeddings.append(embedding)
    #                         queries.append(question)
    #                         # label = index
    #                         labels.append(self.named_labels[index])
    #                 except:
    #                     pass
    #                     # print('queries length:',len(queries))
    #
    #     embeddings = np.array(embeddings)
    #     embeddings = np.squeeze(embeddings)
    #     labels = np.array(labels)
    #
    #     # print (embeddings.shape, labels.shape, len(queries))
    #
    #     return embeddings, labels, queries

    def _prepare_data(self, files):
        print ('prepare data...')

        embeddings = list()
        queries = list()
        queries_ = dict()
        labels = list()
        mlb = MultiLabelBinarizer()

        for index in xrange(len(files)):
            path = files[index]
            label = self.named_labels[index]
            queries_[label] = list()
            with open(path, 'r') as f:
                for line in f:
                    line = json.loads(line.strip().decode('utf-8'))
                    question = line['question']
                    queries_[label].append(question)

        for label, questions in queries_.iteritems():
            for question in questions:
                if question in queries and label not in labels[queries.index(question)]:
                    # print_cn(question)
                    index = queries.index(question)
                    labels[index].append(label)
                else:
                    # print_cn(question)
                    queries.append(question)
                    labels.append([label])
                    tokens = [self.cut(question)]
                    embedding = self.feature_extractor.transform(
                        tokens).toarray()
                    embeddings.append(embedding)

        embeddings = np.array(embeddings)
        embeddings = np.squeeze(embeddings)
        self.mlb = mlb.fit(labels)
        labels = self.mlb.transform(labels)

        # print (embeddings.shape, len(queries))
        # print_cn(labels.shape)

        return embeddings, labels, queries

    def _build(self, files):
        self._build_feature_extractor('tfidf', files)
        return self._prepare_data(files)

    def train(self, pkl, files):
        embeddings, labels, queries = self._build(files)
        print 'train classifier...'

        self.kernel = OneVsRestClassifier(GradientBoostingClassifier(max_depth=5, n_estimators=200))
        self.kernel.fit(embeddings, labels)

        pickle.dump(self, open(pkl, 'wb'))

        print 'train done and saved.'

        # multilabel-indicator is not supported
        # predictions = self.kernel.predict(embeddings)
        # confusion_matrix = metrics.confusion_matrix(predictions, labels)
        # print(confusion_matrix.T)

    def metrics_(self, labels, queries):
        for i in xrange(len(queries)):
            query = queries[i]
            label = labels[i]
            label = np.expand_dims(label, axis=0)
            label = self.mlb.inverse_transform(label)

            label_, probs = self.predict(query)
            label_ = self.mlb.inverse_transform(label_)

            if set(label_[0]) != set(label[0]):
                print_cn(query, [label[0], label_[0], np.squeeze(probs)])

    def validate(self, files):
        embeddings, labels, queries = self._prepare_data(files)
        self.metrics_(labels, queries)

    def predict(self, question):
        line = str(question).replace(" ", "").replace("\t", "")

        embedding = self.feature_extractor.transform(
            [self.cut(line)]).toarray()
        embedding = np.squeeze(embedding)
        embedding = np.reshape(embedding, [1, -1])
        prediction = self.kernel.predict(embedding)
        # label = self.mlb.inverse_transform(prediction)
        probs = self.kernel.predict_proba(embedding)
        return prediction, probs

    def interface(self, q):
        label, probs = self.predict(q)
        probs_dict = {}
        for i in xrange(len(probs[0])):
            probs_dict[self.named_labels[i]] = probs[0][i]
        return self.mlb.inverse_transform(label)[0], probs_dict

    @staticmethod
    def get_instance(path):
        print('loading model file...')
        return pickle.load(open(path, 'r'))


def train():
    clf = SceneClassifier()
    files = ['../../data/supermarket/dialogue.txt', '../../data/supermarket/greetings.txt',
             '../../data/supermarket/qa.txt', ]
    clf.train('../../model/supermarket/scene_clf.pkl', files)


def online_validation():
    clf = SceneClassifier.get_instance('../../model/supermarket/scene_clf.pkl')
    print('loaded model file...')
    try:
        while True:
            question = raw_input('input something...\n')
            prediction, probs = clf.predict(question)
            probs = np.squeeze(probs)
            print 'prediction: {0}, probs: {1}'.format(np.squeeze(clf.mlb.inverse_transform(prediction)), probs)
            print('-------------------------')
    except KeyboardInterrupt:
        print('interaction interrupted')


def offline_validation():
    clf = SceneClassifier.get_instance('../../model/supermarket/scene_clf.pkl')
    print('loaded model file...')
    files = ['../../data/supermarket/dialogue.txt', '../../data/supermarket/greetings.txt',
             '../../data/supermarket/qa.txt', ]
    clf.validate(files)
    # print(clf.interface('我要买鞋'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', choices={'train', 'online', 'offline'},
                        default='offline', help='mode.if not specified,it is in prediction mode')
    args = parser.parse_args()

    if args.m == 'train':
        train()
    elif args.m == 'online':
        online_validation()
    elif args.m == 'offline':
        offline_validation()


if __name__ == '__main__':
    main()
