#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import csv
import jieba
import json
import re

import unicodedata

import _uniout
import cn_util
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from sklearn import metrics
import cPickle as pickle

from query_util import QueryUtils

import argparse


reload(sys)
sys.setdefaultencoding('utf-8')


class SceneClassifier(object):

    def __init__(self):
        self.kernel = None
        # self.embeddings = list()
        self.labels = list()
        self.named_labels = ['business', 'qa',
                             'interaction', 'market', 'repeat_guest', 'repeat_machine']

    def _bulid_ngram(self, files):
        print 'build ngramer...'

        corpus = list()

        for path in files:
            with open(path, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for line in reader:
                    try:
                        line = line[0].replace(" ", "").replace("\t", "")
                        line = QueryUtils.static_remove_cn_punct(line)
                        if line:
                            b = line.encode('utf-8')
                            # print(b)
                            tokens = self.cut(b)

                            corpus.append(tokens)
                    except:
                        pass

        bigram_vectorizer = CountVectorizer(
            ngram_range=(1, 2), min_df=0.0, max_df=1.0, analyzer='char',
            stop_words=[',', '?', '我', '我要', '啊', '呢', '吧'], binary=True)

        self.bigramer = bigram_vectorizer.fit(corpus)

    # def _get_w2v_instance(path):
    #     import gensim
    #     self.w2v = gensim.models.Word2Vec.load('../module/word2vec.bin')

    def cut(self, input_):
        seg = " ".join(jieba.cut(input_, cut_all=False))
        tokens = _uniout.unescape(str(seg), 'utf8')
        return tokens

    # remain to fix...
    # def embedding_by_w2v(self, line):
    #     tokens = self.cut(line)
    #     embedding = []
    #     for word in tokens:
    #         if model.__contains__(word.strip()):
    #             vector = model.__getitem__(word.strip())
    #             result = [v for v in vector]
    #         return result

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
                        line = QueryUtils.static_remove_cn_punct(line)
                        # line = QueryUtils.static_quant_bucket_fix(line)
                        # line = ''.join(line)
                        # print('......')
                        # cn_util.print_cn(line)
                        # line = QueryUtils.quant_bucket_fix(line)
                        # print(line)
                        if line:
                            b = line.encode('utf-8')
                            # print(b)
                            tokens = [self.cut(b)]
                            embedding = self.bigramer.transform(
                                tokens).toarray()
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
        # pre = self.kernel.predict(embeddings)
        # print metrics.confusion_matrix(labels, pre)

        cm = np.zeros((len(self.named_labels), len(self.named_labels)), dtype=np.int32)

        for i in xrange(len(queries)):
            query = queries[i]
            label = labels[i]
            label_, probs = self.predict(query)
            cm[label][self.named_labels.index(label_)] += 1
            if label_ != self.named_labels[label]:
                cn_util.print_cn(query, [self.named_labels[label], label_, probs])
        print(cm)

    def validate(self, files):
        embeddings, labels, tokens = self._prepare_data(files)
        self.metrics_(embeddings, labels, tokens)

    def predict(self, question):
        # clf = pickle.load(open('../model/bqclf.pkl', 'r'))
        line = str(question).replace(" ", "").replace("\t", "")
        b = QueryUtils.static_remove_cn_punct(line)

        fixed = QueryUtils.static_quant_bucket_fix(b)
        fixed = ''.join(fixed)
        # print('......')
        # cn_util.print_cn(fixed)
        # b = ''.join(qu.quant_bucket_fix(b))
        # # cn_util.print_cn('bbbb:', b)

        # b = QueryUtils.static_quant_bucket_fix(b)
        # b = ''.join(b)
        # print('check predict query', b)

        embedding = self.bigramer.transform(
            [self.cut(fixed)]).toarray()
        embedding = np.squeeze(embedding)
        embedding = np.reshape(embedding, [1, -1])
        label = self.kernel.predict(embedding)[0]
        probs = self.kernel.predict_proba(embedding)

        corrected = self.named_labels[self.rule_correct(question, label, probs[0])]

        # print probs
        # print prob
        return corrected, probs

    qa_match_rule = re.compile(r"什么|如何|介绍")

    qa_match_rule = re.compile(ur".*?(什么|如何|介绍|方法|办法|条件|期限).*?")
    interactive_match_rule = re.compile(ur".*?(没有|没啊|对啊|好的|是的|不是|有|好|没|对|是|不).*?")
    def rule_correct(self, q, label_index, probs):

        if label_index == 1:  # qa, correct it to business accordingly
            if not re.match(self.qa_match_rule, q.decode('utf-8')) and probs[label_index] < 0.9:
                return 0
            return label_index
        if label_index == 2 or label_index == 3:
            if re.match(self.interactive_match_rule, q.decode('utf-8')) and probs[label_index] < 0.9:
                return 0

        # if label_index != 0:
        #     if probs[label_index] < 0.6:
        #         return 0
        return label_index

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
    files = ['../data/scene/business_q.txt', '../data/scene/common_qa_q.txt',
             '../data/scene/interactive_g.txt', '../data/scene/market_q.txt', '../data/scene/repeat_guest.txt', '../data/scene/repeat_machine.txt']
    clf.train('../model/scene/sceneclf_six.pkl', files)


def online_validation():
    clf = SceneClassifier.get_instance('../model/scene/sceneclf_six.pkl')
    print('loaded model file...')
    try:
        while True:
            question = raw_input('input something...\n')
            print 'prediction:{0}'.format(clf.predict(question))
    except KeyboardInterrupt:
        print('interaction interrupted')


def offline_validation():
    clf = SceneClassifier.get_instance('../model/scene/sceneclf_six.pkl')
    print('loaded model file...')
    # files = ['../data/scene/business_q.txt', '../data/scene/common_qa_q.txt',
    #          '../data/scene/interactive_g.txt', '../data/scene/market_q.txt', '../data/scene/repeat_guest.txt',
    #          '../data/scene/repeat_machine.txt']
    files = ['../data/scene/test/business.txt', '../data/scene/test/common_qa.txt', '../data/scene/test/interactive.txt',
             '../data/scene/test/market.txt', '../data/scene/test/repeat_guest.txt', '../data/scene/test/repeat_machine.txt']
    clf.validate(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices={'train', 'online_validation', 'offline_validation'},
                        default='train', help='mode.if not specified,it is in prediction mode')
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'online_validation':
        online_validation()
    elif args.mode == 'offline_validation':
        offline_validation()


if __name__ == '__main__':
    main()
