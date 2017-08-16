#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This class is very simple and is stateless
"""
import requests
import random
import jieba

import numpy as np

from sc_qa_clf import SimpleSeqClassifier
from client.sc.sc_belief_clf import Belief_MultiLabel_Clf
from client.query_util import QueryUtils
import client.cn_util

class QAKernel:
    null_anwer = ['啊呀！这可难倒宝宝了！这是十万零一个问题，你要问一下我们对面客服台的客服哥哥姐姐哦！']
    # null_answer = ['null']
    static_clf = None
    def __init__(self):
        print('attaching qa kernel...')
        ## http://localhost:11403/solr/sc_qa/select?fq=entity:%E5%8E%95%E6%89%80&indent=on&q=*:*&wt=json
        self.graph_url = 'http://localhost:11403/solr/graph/select?q.op=OR&wt=json&q=%s'
        self.qa_exact_match_url = 'http://localhost:11403/solr/sc_qa/select?wt=json&q=question:%s'
        if QAKernel.static_clf:
            print('skipping attaching clf kernel...')
            self.clf = QAKernel.static_clf
        else:
            self.clf = SimpleSeqClassifier.get_instance('../../model/sc/qa_clf.pkl')
            QAKernel.static_clf = self.clf

    ## classes: where,whether,when,how,which,what,list
    def kernel(self, q, labels):
        try:
            exact = self.exact_match(q)
            if exact:
                return True, exact
            cls, probs = self.clf.predict(q)
            if cls == 'where':
                success, answer = self.where(q=q)
            if cls == 'whether':
                success, answer = self.whether(q=q, label=labels)
            if cls == 'when':
                success, answer = self.when(q)
            if cls == 'how':
                success, answer = self.how(q)
            if cls == 'which':
                success, answer = self.which(q)
            if cls == 'what':
                success, answer = self.what(q)
            if cls == 'list':
                success, answer = self.list(q)
            return success, answer
        except Exception,e:
            return False, 'mainframe unable to reply since qa/greeting kernel damaged...'

    def common(self, q, key):
        null_answer = "facility has vain %s information" % key
        tokens = ['name:' + s for s in QueryUtils.static_jieba_cut(q, False)]
        q = ' OR '.join(tokens)
        url = self.graph_url % q
        # print('qa_debug:', url)
        # cn_util.print_cn(url)
        r = requests.get(url)

        try:
            num = self._num_answer(r)
            if num > 0:
                response = self._get_response(r=r, key=key, random=True)
                if response:
                    return True, response
                else:
                    return False, null_answer
            else:
                return False, null_answer
        except:
            return False, null_answer

    def where(self, q):
        return self.common(q, 'location')

    def how(self, q):
        return self.common(q, 'application')

    def when(self, q):
        return self.common(q, 'time')

    def what(self, q):
        return self.common(q, 'definition')

    def list(self, q):
        return self.common(q, 'listing')

    ## no logic reasoning
    def which(self, q):
        return self.where(q)

    def whether(self, q, label=None):
        null_answer = "Negative."
        valid_answer = "Affirmative!"
        tokens = ['name:' + s for s in QueryUtils.static_jieba_cut(q, False)]
        q = ' OR '.join(tokens)
        url = self.graph_url % q
        # print('qa_debug:', url)
        # cn_util.print_cn(url)
        r = requests.get(url)

        try:
            num = self._num_answer(r)
            if num > 0:
                if label:
                    labels = self._get_response(r=r, key='label', random=True, keep_array=True)
                    if label in labels:
                        return True, valid_answer
                    else:
                        return True, null_answer
                else:
                    return True, valid_answer
            else:
                return True, null_answer
        except:
            return False, null_answer

    def exact_match(self, q, random_range=1):
        url = self.qa_exact_match_url % q
        r = requests.get(url)
        try:
            num = self._num_answer(r)
            if num > 0:
                x = random.randint(0, min(random_range - 1, num - 1))
                response = self._get_response(r, x)
                return response
            else:
                return None
        except:
            return None

    def _extract_answer(self, r, random_range=1):
        try:
            num = self._num_answer(r)
            if num > 0:
                x = random.randint(0, min(random_range - 1, num - 1))
                response = self._get_response(r, x)
                return True, response
            else:
                return False, np.random.choice(self.null_anwer, 1)[0]
        except:
            return False, np.random.choice(self.null_anwer, 1)[0]

    def _request_solr(self, q):
        ## cut q into tokens
        tokens = ['entity:' + s for s in self.qu.jieba_cut(q, False)]
        q = ' OR '.join(tokens)
        url = self.qa_url % q
        # print('qa_debug:', url)
        # cn_util.print_cn(url)
        r = requests.get(url)
        return r

    def _num_answer(self, r):
        return int(r.json()["response"]["numFound"])

    def _get_response(self, r, key, random=True, keep_array=False):
        try:
            a = r.json()["response"]["docs"][0][key]
            if random:
                rr = np.random.choice(a, 1)[0]
            else:
                if keep_array:
                    return a
                else:
                    rr = ','.join(a)
            return rr.encode('utf8')
        except:
            return None

    def purify_q(self, q):
        q = self.qu.remove_cn_punct(q)
        pos_q = self.qu.corenlp_cut(q, remove_tags=["CD", "VA", "AD", "VC"])
        return ''.join(pos_q), q

if __name__ == '__main__':
    qa = QAKernel()
    client.cn_util.print_cn(qa.kernel(u'LV贵吗'))