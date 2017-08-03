#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" A neural chatbot using sequence to sequence model with
attentional decoder.

This is based on Google Translate Tensorflow model
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/

Sequence to sequence model by Cho et al.(2014)

Created by Chip Huyen as the starter code for assignment 3,
class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu

This file contains the code to build the model

See readme.md for instruction on how to run the starter code.
"""
from __future__ import print_function

import time
import csv
import sys
import numpy as np
import tensorflow as tf
import inspect
import jieba
import _uniout
import cn_util
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer


from sklearn.naive_bayes import MultinomialNB
import cPickle as pickle

reload(sys)
sys.setdefaultencoding("utf-8")


class SeqClassifier:

    def __init__(self, data_path):
        print('initilizing classifier...')
        self.data_path = data_path
        self.num_vol = 0
        self.vol = {}
        self.classes = {}
        self.index_classes = {}
        self.classes_num_sub = {}
        self.classifiers = {}

    def _ngram(self):
        corpus = list()
        with open(self.data_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for line in reader:
                # print('line:', line)
                b = line[1].encode('utf-8')
                # print(b)
                tokens = self.cut(b)

                corpus.append(tokens)

        bigram_vectorizer = CountVectorizer(
            ngram_range=(1, 2), min_df=0.0, max_df=1.0, analyzer='char', stop_words=[',', '?', '我', '我要'], binary=True)

        self.bigramer = bigram_vectorizer.fit(corpus)

    def build(self):
        self._ngram()
        index = 0
        with open(self.data_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for line in reader:
                index = index + 1

                a = line[0].encode('utf-8')
                b = line[1].encode('utf-8')
                last_slot, slot = a.split(",")
                # print("a:{},b:{}".format(a, b))
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

                if last_slot != "ROOT":
                    ax = self.cut(last_slot)
                    self.into_vol(ax)

                ax = self.cut(slot)

                self.into_vol(ax)

                bx = self.cut(b)
                # print(bx)
                self.into_vol(bx)

    def train_classifier(self):
        embeddings = {}
        classes = {}
        weights = {}
        with open(self.data_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for line in reader:
                key = line[0].encode('utf-8')
                input_ = line[1].encode('utf-8')
                last_slot, slot = key.split(',')
                tokens = [self.cut(input_)]
                embedding = self.bigramer.transform(tokens).toarray()
                # print(embedding[0])
                if last_slot not in embeddings:
                    embeddings[last_slot] = []
                embeddings[last_slot].append(embedding[0])

                # print('last_slot:', last_slot)
                if last_slot not in classes:
                    # print('class:', last_slot)
                    classes[last_slot] = []
                cls = self.classes[last_slot][slot]
                classes[last_slot].append(cls)

                # if last_slot not in weights:
                #     weights[last_slot] = []
                # w = float(line[2])
                # weights[last_slot].append(w)

        # for key, embs in embeddings.iteritems():
        #     embeddings[key] = np.array(embs)
        for key, cs in classes.iteritems():
            classes[key] = np.array(cs)
        for key, ww in weights.iteritems():
            weights[key] = np.array(ww)

        for i, last_slot in enumerate(classes.keys()):
            print("training classifier", i, last_slot)
            if self.classes_num_sub[last_slot] > 1:
                clf = GradientBoostingClassifier(max_depth=5, n_estimators=200)
                # clf = MultinomialNB(
                #     alpha=0.01, class_prior=None, fit_prior=True)
                clf.fit(embeddings[last_slot], classes[
                        last_slot])
                self.classifiers[last_slot] = clf

        # test
        # input_ = '办卡'
        # print(self.predict('ROOT', input_))

    def into_vol(self, tokens):
        _tokens = tokens.split(" ")
        for token in _tokens:
            if token == "," or token == "?" or token == "":
                continue
            if token not in self.vol:
                self.vol[token] = self.num_vol
                self.num_vol = self.num_vol + 1

    def sequence_to_embedding(self, tokens):
        vector = np.zeros(self.num_vol)
        _tokens = tokens.split(" ")
        for token in _tokens:
            if token == ',' or token == '?' or token == '':
                continue
            index = self.vol[token]
            vector[index] = 1
        return vector

    def cut(self, input_):
        # return self.tokenize(input_)
        return self.jieba_cut(input_)

    def jieba_cut(self, input_):
        seg = " ".join(jieba.cut(input_, cut_all=False))
        tokens = _uniout.unescape(str(seg), 'utf8')
        return tokens

    def tokenize(self, input_):
        s = input_
        L = []
        for ch in s:
            L.append(ch)

        return " ".join(L)

    def predict(self, parent_slot, input_):
        if self.classes_num_sub[parent_slot] == 1:
            for cls, num in self.classes[parent_slot].iteritems():
                return cls, 1.0
        tokens = [self.cut(input_)]
        # print('jieba_cut:', _uniout.unescape(str(tokens), 'utf8'))
        embeddings = np.reshape(
            self.bigramer.transform(tokens).toarray()[0], [1, -1])
        clf = self.classifiers[parent_slot]
        class_ = clf.predict(embeddings)
        probs = clf.predict_proba(embeddings)
        for c in class_:
            # c - 1 as the 1st class is 1 not zero
            return self.index_classes[parent_slot][c], probs[0][c - 1]

    def test(self, model_path):
        correct = 0.0
        total = 0.0
        with open(model_path, "rb") as input_file:
            # test
            with open(self.data_path, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for line in reader:
                    try:
                        key = line[0].encode('utf-8')
                        input_ = line[1].encode('utf-8')
                        last_slot, slot = key.split(',')
                        prediction, proba = self.predict(last_slot, input_)
                        if prediction == slot:
                            correct = correct + 1
                        else:
                            print(input_, last_slot, slot, prediction)
                        total = total + 1
                    except Exception, e:
                        print(e.message)
        print('accuracy:' + str(correct / total))


if __name__ == "__main__":
    clf = SeqClassifier("../../data/sc/pruned_dialogue.txt")
    clf.build()
    clf.train_classifier()
    with open("../../model/ss_clf.pkl", 'wb') as pickle_file:
        pickle.dump(clf, pickle_file, pickle.HIGHEST_PROTOCOL)

    with open("../../model/ss_clf.pkl", "rb") as input_file:
        _clf = pickle.load(input_file)
        # input_ = '取'
        # print(cn_util.cn(_clf.predict('取款两万以下', input_)))
        # input_ = '取两百不用银行卡'
        # print(cn_util.cn(_clf.predict('ROOT', input_)))

        _clf.test("../../data/sc/pruned_dialogue.txt")

        # print("self.classes:", _uniout.unescape(str(self.classes), 'utf-8'))
        # print('************************************************************')
        # print("self.index_classes:", _uniout.unescape(
        #     str(self.helper.index_classes), 'utf-8'))
        # print('************************************************************')
        # print("self.classes_num_sub:", _uniout.unescape(
        #     str(self.classes_num_sub), 'utf-8'))
