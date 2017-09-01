#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import jieba
import gensim
import os
import requests
import traceback

import _uniout
from cn_util import print_cn
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
import cPickle as pickle

import fasttext

import argparse

# from sc_scene_clf import SceneClassifier
from query_util import QueryUtils

reload(sys)
sys.setdefaultencoding('utf-8')

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

# w2v_model = gensim.models.Word2Vec.load('/media/deep/DATA1/w2v/w2c_cb')


class SceneClassifier:
    def __init__(self):
        self.kernel = None
        self.named_labels = ['base','greeting','qa','repeat_user','repeat_machine','sale']
        self.fasttext_url = "http://localhost:11425/fasttext/s2v?q={0}&w="
        self.fasttext_url_weighted = "http://localhost:11425/fasttext/s2v?q={0}&w={1}"
        self.weighted = False

    def _add_extra_dict(self, path):
        with open(path, 'r') as inp:
            for line in inp:
                line = line.split(':')[-1]
                words = line.split(',')
                for word in words:
                    jieba.add_word(word)

    def cut(self, input_):
        input_ = QueryUtils.static_remove_cn_punct(input_)
        seg = " ".join(jieba.cut(input_, cut_all=False))
        tokens = _uniout.unescape(str(seg), 'utf8')
        return tokens

    def get_w2v_emb(self,tokens):
        # embedding=np.zeros((1,300),dtype=np.float32)
        # count=0
        # # print_cn(tokens)
        # for word in tokens:
        #     word = word.encode('utf-8')
        #     if w2v_model.__contains__(word.strip()):
        #         vector = w2v_model.__getitem__(word.strip())
        #         result = [v for v in vector]
        #
        #         embedding=np.add(embedding,np.asarray(result))
        #         # print embedding
        #         count+=1
        # if count==0:
        #     print('get...',count)
        #     print_cn(tokens)
        # embedding=np.divide(embedding,count)
        ## get fasttext embedding from web

        embedding = self._fasttext_vector(tokens)
        return np.squeeze(embedding)

    def _fasttext_vector(self, tokens):
        if not self.weighted:
            try:
                weights = np.ones(shape=len(tokens))
                url = self.fasttext_url_weighted.format(','.join(tokens), ",".join([str(weight) for weight in weights]))
            except:
                traceback.print_exc()
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
        try:
            r = requests.get(url=url)
            vector = r.json()['vector']
            return vector
        except:
            print_cn(url)
            traceback.print_exc()
            return None

    # def check_zero_tokens(self,tokens):
    #     count=0
    #     for word in tokens:
    #         word = word.encode('utf-8')
    #         if w2v_model.__contains__(word.strip()):
    #             count+=1
    #     if count==0:
    #         print_cn(tokens)
    #
    #     return True if count!=0 else False

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
                    # line = json.loads(line.strip().decode('utf-8'))
                    # question = line['question']
                    question = line.replace('\t','').replace(' ','').strip('\n').decode('utf-8')
                    question = QueryUtils.static_remove_cn_punct(str(question))
                    tokens= QueryUtils.static_jieba_cut(question)
                    # print_cn(tokens)
                    if len(tokens)==0:
                        continue
                    # cc=self.check_zero_tokens(tokens)
                    # if not cc:
                    #     continue
                    queries_[label].append(question)
        # print len(queries_)
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
                    tokens = self.cut(question).split(' ')
                    embedding=self.get_w2v_emb(tokens)
                    embeddings.append(embedding)

        embeddings = np.array(embeddings)
        embeddings = np.squeeze(embeddings)
        self.mlb = mlb.fit(labels)
        labels = self.mlb.transform(labels)

        # print (embeddings.shape, len(queries))
        # print_cn(labels.shape)

        return embeddings, labels, queries

    def _build(self, files):
        self._add_extra_dict('../data/sc/belief_graph.txt')
        return self._prepare_data(files)

    def train(self, pkl, files):
        embeddings, labels, queries = self._build(files)
        print 'train classifier...'

        self.kernel = OneVsRestClassifier(GradientBoostingClassifier(max_depth=5, n_estimators=1000))
        self.kernel.fit(embeddings, labels)

        pickle.dump(self, open(pkl, 'wb'))

        print 'train done and saved.'

        print 'validation...'
        self.metrics_(labels, queries)


    def metrics_(self, labels, queries):
        correct = 0.0
        total = 0
        for i in xrange(len(queries)):
            query = queries[i]
            if not query:
                continue
            total += 1
            label = labels[i]
            label = np.expand_dims(label, axis=0)
            real = self.mlb.inverse_transform(label)[0]
            real = list(real)
            label_, probs = self.predict(query)
            label_ = list(set(label_))
            # label_ = self.mlb.inverse_transform(label_)

            if ' '.join(real) != ' '.join(list(label_)):
                print('{0}: {1}-->{2}'.format(query, ' '.join(real), ' '.join(list(label_))))
            else:
                correct += 1
        print('accuracy:{0}'.format(correct / total))

    def validate(self, files):
        embeddings, labels, queries = self._prepare_data(files)
        self.metrics_(labels, queries)

    def predict(self, question):
        line = str(question).replace(" ", "").replace("\t", "")
        tokens=self.cut(line).split(' ')
        embedding=self.get_w2v_emb(tokens)
        embedding = np.reshape(embedding, [1, -1])
        prediction = self.kernel.predict(embedding)
        prediction_index_first_sample = np.where(prediction[0] == 1)
        # label = self.mlb.inverse_transform(prediction)
        probs = self.kernel.predict_proba(embedding)
        ## note that in prediction stage n_sample==1
        label_ = self.mlb.inverse_transform(prediction)
        if len(label_[0]) == 0:
            index = np.argmax(probs[0])
            l = self.named_labels[index]
            prob = probs[0][index]
            return [l], [prob]
        return label_[0], probs[0][prediction_index_first_sample]

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
    files = ['../data/sc/scene/base', '../data/sc/scene/greeting',
             '../data/sc/scene/qa', '../data/sc/scene/repeat_guest',
             '../data/sc/scene/repeat_machine','../data/sc/scene/sale']
    clf.train('../model/sc/scene_embeded_clf.pkl', files)


def online_validation():
    clf = SceneClassifier.get_instance('../model/sc/scene_embeded_clf.pkl')
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
    clf = SceneClassifier.get_instance('../model/sc/scene_embeded_clf.pkl')
    print('loaded model file...')
    files = ['../data/sc/scene/base', '../data/sc/scene/greeting',
             '../data/sc/scene/qa', '../data/sc/scene/repeat_guest',
             '../data/sc/scene/repeat_machine', '../data/sc/scene/sale']
    clf.validate(files)
    # print(clf.interface('我要买鞋'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', choices={'train', 'online', 'offline'},
                        default='online', help='mode.if not specified,it is in prediction mode')
    args = parser.parse_args()

    if args.m == 'train':
        train()
    elif args.m == 'online':
        online_validation()
    elif args.m == 'offline':
        offline_validation()

# def test():
#     clf = SceneClassifier()
#     files = ['../data/sc/scene/base', '../data/sc/scene/greeting',
#              '../data/sc/scene/qa', '../data/sc/scene/repeat_guest',
#              '../data/sc/scene/repeat_machine', '../data/sc/scene/sale', '../data/sc/scene/negative']
#
#     embeddings, labels, queries = clf._prepare_data(files)

if __name__ == '__main__':
    main()