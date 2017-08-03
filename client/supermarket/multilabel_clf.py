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
from cn_util import print_cn

reload(sys)
sys.setdefaultencoding("utf-8")


class Multilabel_Clf():
    def __init__(self, data_path):
        self.data_path = data_path
        self.classes = {}
        self.mlbs = {}
        self.index_classes = {}
        self.classes_num_sub = {}
        self.classifiers = {}
        self.build()

    def _ngram(self, data_path):
        print('Build ngram...')
        corpus = list()
        with open(data_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for line in reader:
                b = line[1].encode('utf-8')
                tokens = self.cut(b)
                corpus.append(tokens)

        bigram_vectorizer = CountVectorizer(
            ngram_range=(1, 2), min_df=0.0, max_df=1.0, analyzer='char', stop_words=[',', '?', '我', '我要'], binary=True)
        self.bigramer = bigram_vectorizer.fit(corpus)

    def _build_feature_extraction(self, mode, data_path):
        print('Build feature extraction...')
        corpus = list()
        with open(data_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for line in reader:
                b = line[1].encode('utf-8')
                tokens = self.cut(b)
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

    def build(self):
        self._build_feature_extraction('tfidf', self.data_path)
        index = 0
        with open(self.data_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for line in reader:
                index = index + 1
                a = line[0].encode('utf-8')
                last_slot, slot = a.split(",")
                if last_slot not in self.classes:
                    self.classes[last_slot] = {}
                    self.index_classes[last_slot] = {}
                    self.classes_num_sub[last_slot] = 1
                    self.classes.get(last_slot)[slot] = 1
                    self.index_classes.get(last_slot)[1] = slot
                else:
                    if slot not in self.classes.get(last_slot):
                        self.classes.get(last_slot)[
                            slot] = self.classes_num_sub.get(last_slot) + 1
                        self.index_classes.get(last_slot)[
                            self.classes_num_sub.get(last_slot) + 1] = slot
                        self.classes_num_sub[
                            last_slot] = self.classes_num_sub.get(last_slot) + 1

        for key in self.classes.keys():
            labels = list()
            mlb = MultiLabelBinarizer()
            # print_cn(key)
            for k in self.classes[key].keys():
                label = list()
                label.append(key)
                label.append(k)
                labels.append(label)
            # print_cn(labels)
            self.mlbs[key] = mlb.fit(labels)

        print('Build done.')

    def train(self):
        embeddings = {}
        classes = {}

        with open(self.data_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for line in reader:
                key = line[0].encode('utf-8')
                input_ = line[1].encode('utf-8')
                last_slot, slot = key.split(',')
                tokens = [self.cut(input_)]
                embedding = self.feature_extractor.transform(tokens).toarray()

                if last_slot not in embeddings:
                    embeddings[last_slot] = []
                embeddings[last_slot].append(embedding[0])

                if last_slot not in classes:
                    classes[last_slot] = []
                classes[last_slot].append([last_slot, slot])

        for i, last_slot in enumerate(classes.keys()):
            print("training classifier", i + 1, last_slot)
            if self.classes_num_sub[last_slot] > 1:
                clf = OneVsRestClassifier(GradientBoostingClassifier(max_depth=5, n_estimators=200))
                X = np.array(embeddings[last_slot])
                # print_cn(classes[last_slot])
                y = self.mlbs[last_slot].transform(classes[last_slot])
                clf.fit(X, y)
                self.classifiers[last_slot] = clf

        print('Train done.')

    def predict(self, parent_slot, input_):
        if self.classes_num_sub[parent_slot] == 1:
            for cls, num in self.classes[parent_slot].iteritems():
                return cls, 1.0
        tokens = [self.cut(input_)]
        # print('jieba_cut:', _uniout.unescape(str(tokens), 'utf8'))
        embeddings = np.reshape(
            self.feature_extractor.transform(tokens).toarray()[0], [1, -1])
        clf = self.classifiers[parent_slot]
        mlb = self.mlbs[parent_slot]
        prediction = clf.predict(embeddings)
        print('prediction:',prediction)
        labels = mlb.inverse_transform(prediction)
        # print_cn(labels)
        probs = clf.predict_proba(embeddings)

        return labels[0], probs

    def test(self, test_path):
        correct = 0.0
        total = 0.0

        with open(test_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for line in reader:
                # print_cn(line)
                key = line[0].encode('utf-8')
                input_ = line[1].encode('utf-8')
                last_slot, slot = key.split(',')
                real = [last_slot, slot]
                prediction, proba = self.predict(last_slot, input_)
                if set(prediction) == set([last_slot, slot]):
                    correct = correct + 1
                else:
                    print('{0}: {1}-->{2}'.format(input_, ' '.join(real), ' '.join(prediction)))
                total = total + 1
        print('accuracy:{0}'.format(correct / total))


def train(train_data_path, model_path):
    clf = Multilabel_Clf(train_data_path)
    clf.train()
    with open(model_path, 'wb') as pickle_file:
        pickle.dump(clf, pickle_file, pickle.HIGHEST_PROTOCOL)


def test(test_data_path, model_path):
    with open(model_path, "rb") as input_file:
        clf = pickle.load(input_file)
        clf.test(test_data_path)


def main():
    model_path = '../../model/supermarket/multilabel_clf.pkl'
    train_data_path = '../../data/supermarket/pruned_dialogue.txt'
    test_data_path = '../../data/supermarket/pruned_dialogue.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', choices={'train', 'test'},
                        default='test', help='mode.if not specified,it is in test mode')

    args = parser.parse_args()

    if args.m == 'train':
        train(train_data_path, model_path)
    elif args.m == 'test':
        test(test_data_path, model_path)
    else:
        print('Unknow mode, exit.')


if __name__ == '__main__':
    main()
