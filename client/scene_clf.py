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
        # self.embeddings = list()
        self.labels = list()
        self.named_labels = ['business', 'qa', 'interaction','market']

    def _bulid_ngram(self, files):
        print 'build ngramer...'

        corpus = list()

        for path in files:
            with open(path, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for line in reader:
                    try:
                        line = line[0].replace(" ", "").replace("\t", "")
                        if line:
                            b = line.encode('utf-8')
                            # print(b)
                            tokens = self.cut(b)

                            corpus.append(tokens)
                    except:
                        pass

        bigram_vectorizer = CountVectorizer(
            ngram_range=(1, 2), min_df=0.0, max_df=1.0, analyzer='char',
            stop_words=[',', '?', '我', '我要','啊','呢','吧'], binary=True)

        self.bigramer = bigram_vectorizer.fit(corpus)

    def cut(self, input_):
        seg = " ".join(jieba.cut(input_, cut_all=False))
        tokens = _uniout.unescape(str(seg), 'utf8')
        return tokens

    def _prepare_data(self, files):
        print 'prepare data...'

        embeddings = list()
        queries = list()
        labels = list()

        for index in xrange(len(files)):
            path = files[index]
            with open(path, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for line in reader:
                    try:
                        line = line[0].replace(" ", "").replace("\t", "")
                        if line:
                            b = line.encode('utf-8')
                            # print(b)
                            tokens = [self.cut(b)]
                            embedding = self.bigramer.transform(tokens).toarray()
                            embeddings.append(embedding)
                            queries.append(b)
                            label = index

                            labels.append(label)
                    except:
                        pass

        embeddings = np.array(embeddings)
        embeddings = np.squeeze(embeddings)

        labels = np.array(labels)

        # embeddings, labels = shuffle(
        #     embeddings, labels, random_state=0)

        return embeddings, labels, queries

    def _build(self, files):
        self._bulid_ngram(files)
        return self._prepare_data(files)

    def train(self, pkl, files):
        embeddings, labels, queries = self._build(files)
        print 'train classifier...'

        self.kernel = GradientBoostingClassifier(max_depth=5, n_estimators=200)
        self.kernel.fit(embeddings, labels)

        pickle.dump(self, open(pkl, 'wb'))

        print 'train done and saved.'
        self.metrics_(embeddings, labels, queries)

    def metrics_(self, embeddings, labels, queries):
        line = "取款"
        print(self.predict(line))
        pre = self.kernel.predict(embeddings)
        print metrics.confusion_matrix(labels, pre)

        for i in xrange(len(queries)):
            query = queries[i]
            label = labels[i]
            label_, probs = self.predict(query)
            if label_ != self.named_labels[label]:
                cn_util.print_cn(query, [self.named_labels[label], label_])

        # precision_score = metrics.precision_score(self.labels, pre)
        # recall_score = metrics.recall_score(self.labels, pre)
        # f1_score = metrics.f1_score(self.labels, pre)

        # print 'precision_score:{0}, recall_score:{1},
        # f1_score:{2}'.format(precision_score, recall_score, f1_score)

    def find_wrong(self):
        pass

    def validate(self, files):
        embeddings, labels, tokens = self._build(files)
        self.metrics_(embeddings, labels, tokens)

    def predict(self, question):
        # clf = pickle.load(open('../model/bqclf.pkl', 'r'))
        line = str(question).strip()
        b = line.encode('utf-8')
        embedding = self.bigramer.transform(
            [self.cut(b)]).toarray()
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
        for i in xrange(len(probs[0])):
            probs_dict[self.named_labels[i]] = probs[0][i]
        return label, probs_dict

    @staticmethod
    def get_instance(path):
        print('loading model file...')
        return pickle.load(open(path, 'r'))


def train():
    clf = SceneClassifier()
    files = ['../data/scene/business_q.txt','../data/scene/common_qa_q.txt','../data/scene/interactive_g.txt', '../data/scene/market_q.txt']
    clf.train('../model/scene/sceneclf.pkl', files)
    # clf.train('../model/scene/sceneclf.pkl', '../data/scene/a.txt', '../data/scene/b.txt',
              # '../data/scene/c.txt')

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

def offline_validation():
    clf = SceneClassifier.get_instance('../model/scene/sceneclf.pkl')
    print('loaded model file...')
    files = ['../data/scene/business_q.txt', '../data/scene/common_qa_q.txt', '../data/scene/interactive_g.txt',
             '../data/scene/market_q.txt']
    clf.validate(files)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices={'train', 'online_validation', 'offline_validation'},
                        default='online_validation', help='mode.if not specified,it is in prediction mode')
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'online_validation':
        online_validation()
    elif args.mode == 'offline_validation':
        offline_validation()


if __name__ == '__main__':
    main()
