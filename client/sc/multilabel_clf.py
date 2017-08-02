#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
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

reload(sys)
sys.setdefaultencoding("utf-8")



class Multilabel_Clf():

    def __init__(self,corpus_path,pruned_path,train_data_path,test_data_path,model_path):
        self.corpus_path=corpus_path
        self.pruned_path=pruned_path
        self.train_data_path=train_data_path
        self.test_data_path=test_data_path
        self.model_path=model_path
        self.build()


    def _maybe_process(self,inpath, outpath):
        if os.path.exists(outpath):
            print('Data already existe.')
        else:
            with open(outpath, 'w+') as out:
                with open(inpath, 'r') as inp:
                    for l in inp:
                        result = dict()
                        result['label'] = list()
                        line = json.loads(l.strip().decode('utf-8'))
                        result['X'] = line['question']
                        result['label'].append(line['intention'])
                        if line['type'] not in result['label']:
                            result['label'].append(line['type'])
                        json.dump(result, out, ensure_ascii=False)
                        out.write('\n')
            print('Data process done.')

    def _ngram(self,pruned_path):
        print('Build ngram...')
        corpus = list()
        with open(pruned_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for line in reader:
                b = line[1].encode('utf-8')
                tokens = self.cut(b)
                corpus.append(tokens)

        bigram_vectorizer = CountVectorizer(
            ngram_range=(1, 2), min_df=0.0, max_df=1.0, analyzer='char', stop_words=[',', '?', '我', '我要'], binary=True)
        self.bigramer = bigram_vectorizer.fit(corpus)

    def cut(self, input_):
        seg = " ".join(jieba.cut(input_, cut_all=False))
        tokens = _uniout.unescape(str(seg), 'utf8')
        return tokens


    def _embedding(self,train_data_path):
        print('Build embedding...')
        self.mlb = MultiLabelBinarizer()
        embeddings=list()
        labels=list()

        with open(train_data_path,'r') as inp:
            for l in inp:
                line=json.loads(l.strip().decode('utf-8'))
                X=line['X']
                label=line['label']
                tokens = [self.cut(X)]
                embedding = self.bigramer.transform(tokens).toarray()
                embeddings.append(embedding[0])
                labels.append(label)

        self.embeddings=np.array(embeddings)
        self.labels=self.mlb.fit_transform(labels)

    def build(self):
        self._maybe_process(self.corpus_path,self.train_data_path)
        self._ngram(self.pruned_path)
        self._embedding(self.train_data_path)
        print('Build done.')


    def train(self):
        print('Start training classifier...')
        self.gbdt_clf=OneVsRestClassifier(GradientBoostingClassifier(max_depth=5, n_estimators=200))
        # print (self.embeddings.shape,self.labels.shape)
        self.gbdt_clf.fit(self.embeddings,self.labels)
        pickle.dump(self.gbdt_clf, open(self.model_path, 'wb'))
        print('Train done and saved.')

    def prediction(self):
        print('loaded model file...')
        gbdt_clf = pickle.load(open(self.model_path, 'rb'))

        try:
            while True:
                input_ = raw_input('input something...\n')
                tokens = [self.cut(input_)]
                embedding = np.reshape(
                    self.bigramer.transform(tokens).toarray()[0], [1, -1])
                prediction = gbdt_clf.predict(embedding)
                labels = self.mlb.inverse_transform(prediction)

                print('{0} --> {1}'.format(input_, ' '.join(labels[0])))
                print('-----------------------------')
        except KeyboardInterrupt:
            print('interaction interrupted')

    def predict(self, question, parent_class):
        return None

    def test(self):
        questions=list()
        embeddings = list()
        labels = list()
        gbdt_clf = pickle.load(open(self.model_path, 'rb'))

        with open(self.test_data_path, 'r') as inp:
            for l in inp:
                line = json.loads(l.strip().decode('utf-8'))
                question = line['X']
                questions.append(question)
                label = line['label']
                tokens = [self.cut(question)]
                embedding = self.bigramer.transform(tokens).toarray()
                embeddings.append(embedding[0])
                labels.append(label)
        embeddings = np.array(embeddings)
        ft_labels = self.mlb.fit_transform(labels)
        predictions=gbdt_clf.predict(embeddings)
        print ('Accuracy score:',accuracy_score(ft_labels,predictions))

        invers_pred=self.mlb.inverse_transform(predictions)
        for question, real, pred in zip(questions,labels,invers_pred):
            if set(real)!=set(pred):
                print('{0}: {1}-->{2}'.format(question,' '.join(real),' '.join(pred)))



def train(clf):
    clf.train()

def online_validation(clf):
    clf.prediction()

def offline_validation(clf):
    clf.test()

def main():
    clf = Multilabel_Clf('../../data/supermarket/dialogue.txt', '../../data/supermarket/pruned_dialogue.txt', \
                         '../../data/supermarket/train.txt', '../../data/supermarket/train.txt',
                         '../../model/sm_multilabel_clf.pkl')

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices={'train', 'online_validation', 'offline_validation'},
                        default='offline_validation', help='mode.if not specified,it is in prediction mode')
    args = parser.parse_args()

    if args.mode == 'train':
        train(clf)
    elif args.mode == 'online_validation':
        online_validation(clf)
    elif args.mode == 'offline_validation':
        offline_validation(clf)

if __name__ == '__main__':
    main()
