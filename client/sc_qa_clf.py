#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import jieba
import json
import os
import re

import _uniout
from client.cn_util import print_cn
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
import cPickle as pickle

import argparse

# from sc_scene_clf import SceneClassifier
from query_util import QueryUtils

reload(sys)
sys.setdefaultencoding('utf-8')

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)


class SimpleSeqClassifier:
    def __init__(self):
        self.kernel = GradientBoostingClassifier(max_depth=5, n_estimators=200)
        self.named_labels = ["where","exist","whether","ask_price","ask_discount","ask_queue","permit","which","what","how","list","when"]
        # self.qu = QueryUtils()

    def _build_feature_extractor(self, mode, files):
        print('Build feature extraction...')
        corpus = list()

        for path in files:
            with open(path, 'r') as f:
                for line in f:
                    # line = json.loads(line.strip().decode('utf-8'))
                    # question = line['question']
                    question = line.replace('\t', '').replace(' ', '').strip('\n').decode('utf-8')
                    question = question.split('#')[0]
                    # question = QueryUtils.static_remove_cn_punct(str(question))
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
        input_ = QueryUtils.static_simple_remove_punct(input_)
        seg = " ".join(jieba.cut(input_, cut_all=False))
        tokens = _uniout.unescape(str(seg), 'utf8')
        return tokens

    def _prepare_data(self, files):
        print ('prepare data...')

        embeddings = list()
        queries = list()
        labels = list()
        # mlb = MultiLabelBinarizer()

        for index in xrange(len(files)):
            path = files[index]
            with open(path, 'r') as f:
                for line in f:
                    # line = json.loads(line.strip().decode('utf-8'))
                    # question = line['question']
                    line = line.replace('\t','').replace(' ','').strip('\n').decode('utf-8').split('#')
                    question = QueryUtils.static_simple_remove_punct(str(line[0]))
                    label = self.named_labels.index(str(line[1].encode('utf-8')))
                    queries.append(question)
                    labels.append(label)
                    tokens = [self.cut(question)]
                    embedding = self.feature_extractor.transform(
                        tokens).toarray()
                    embeddings.append(embedding)

        embeddings = np.array(embeddings)
        embeddings = np.squeeze(embeddings)
        # self.kernel.fit()
        # self.mlb = mlb.fit(labels)
        # labels = self.mlb.transform(labels)

        # print (embeddings.shape, len(queries))
        # print_cn(labels.shape)

        return embeddings, labels, queries

    def _build(self, files):
        self._build_feature_extractor('tfidf', files)
        return self._prepare_data(files)

    def train(self, pkl, files):
        embeddings, labels, queries = self._build(files)
        print 'train classifier...'

        self.kernel = GradientBoostingClassifier(max_depth=5, n_estimators=200)
        self.kernel.fit(embeddings, labels)

        pickle.dump(self, open(pkl, 'wb'))

        print 'train done and saved.'
        self.metrics_(labels, queries, embeddings)

    def metrics_(self, labels, queries, embeddings):

        predictions = self.kernel.predict(embeddings)
        confusion_matrix_ = confusion_matrix(predictions, labels)
        print(confusion_matrix_.T)

        correct = 0.0
        total = 0

        for i in xrange(len(queries)):
            query = queries[i]
            if not query:
                continue
            total += 1
            real = labels[i]
            # real = self.mlb.inverse_transform(label)[0]
            # real = list(real)
            label_, probs = self.predict(query)
            label_ = self.named_labels.index(label_)
            # label_ = self.mlb.inverse_transform(label_)

            if real != label_:
                print('{0}: {1}-->{2}'.format(query, real, label_))
            else:
                correct += 1
        print('accuracy:{0}'.format(correct / total))

    def validate(self, files):
        embeddings, labels, queries = self._prepare_data(files)
        self.metrics_(labels, queries)

    def predict(self, question):
        line = str(question).replace(" ", "").replace("\t", "")

        embedding = self.feature_extractor.transform(
            [self.cut(line)]).toarray()
        embedding = np.squeeze(embedding)
        embedding = np.reshape(embedding, [1, -1])
        label = self.kernel.predict(embedding)[0]
        probs = self.kernel.predict_proba(embedding)
        # print probs
        # print prob
        corrected = self.correct_label(question.decode('utf-8'), self.named_labels[label])
        return corrected, probs

    ask_price = re.compile(ur'.*?(贵吗|贵不贵|便宜吗|便宜不便宜).*?')
    list = re.compile(ur'.*?(有什么|有哪些).*?')
    ask_discount = re.compile(ur'.*?(优惠吗|折扣吗|有没有优惠|有没有折扣).*?')
    where = re.compile(ur'.*?(几楼|什么地方|在哪|在那|带我去|我想去|怎么去|怎么走).*?')
    def correct_label(self, question, label):
        if re.match(self.ask_price, question):
            return 'ask_price'
        if re.match(self.ask_discount, question):
            return 'ask_discount'
        if re.match(self.ask_discount, question):
            return 'where'
        if re.match(self.list, question):
            return 'list'
        return label

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
    clf = SimpleSeqClassifier()
    files = ['../data/sc/train/pruned_qa.txt']
    clf.train('../model/sc/qa_clf.pkl', files)


def online_validation():
    clf = SimpleSeqClassifier.get_instance('../model/sc/qa_clf.pkl')
    print('loaded model file...')
    try:
        while True:
            question = raw_input('input something...\n')
            prediction, probs = clf.predict(question)
            probs = np.squeeze(probs)
            print 'prediction: {0}, probs: {1}'.format(np.squeeze(prediction), probs)
            print('-------------------------')
    except KeyboardInterrupt:
        print('interaction interrupted')


def offline_validation():
    clf = SimpleSeqClassifier.get_instance('../model/sc/scene_clf.pkl')
    print('loaded model file...')
    files = ['../data/sc/train/pruned_qa.txt']
    clf.validate(files)
    # print(clf.interface('我要买鞋'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', choices={'train', 'online', 'offline'},
                        default='train', help='mode.if not specified,it is in prediction mode')
    args = parser.parse_args()

    if args.m == 'train':
        train()
    elif args.m == 'online':
        online_validation()
    elif args.m == 'offline':
        offline_validation()


if __name__ == '__main__':
    main()