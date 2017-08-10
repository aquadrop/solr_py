#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This class is very simple and is stateless
"""
import requests
import random
import jieba

import numpy as np

from query_util import QueryUtils
import cn_util

class QAKernel:
    null_anwer = ['啊呀！这可难倒宝宝了！这是十万零一个问题，你要问一下我们对面客服台的客服哥哥姐姐哦！']
    # null_answer = ['null']

    def __init__(self):
        print('attaching qa kernel...')
        ## http://localhost:11403/solr/sc_qa/select?fq=entity:%E5%8E%95%E6%89%80&indent=on&q=*:*&wt=json
        self.qa_url = 'http://localhost:11403/solr/sc_qa/select?q.op=OR&wt=json&q=%s'
        self.qa_exact_match_url = 'http://localhost:11403/solr/sc_qa/select?wt=json&q=question:%s'
        self.qu = QueryUtils()

    def kernel(self, q):
        try:
            exact = self.exact_match(q)
            if exact:
                return exact
            r = self._request_solr(q)
            answer = self._extract_answer(r)
            return answer
        except Exception,e:
            return 'mainframe unable to reply since qa/greeting kernel damaged...'

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
                return response
            else:
                return np.random.choice(self.null_anwer, 1)[0]
        except:
            return np.random.choice(self.null_anwer, 1)[0]

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

    def _get_response(self, r, i = 0):
        try:
            a = r.json()["response"]["docs"][i]['answer']
            rr = np.random.choice(a, 1)[0]
            x = random.randint(0, min(0, len(a) - 1))
            return rr.encode('utf8')
        except:
            return None

    def purify_q(self, q):
        q = self.qu.remove_cn_punct(q)
        pos_q = self.qu.corenlp_cut(q, remove_tags=["CD", "VA", "AD", "VC"])
        return ''.join(pos_q), q

if __name__ == '__main__':
    qa = QAKernel()
    print(qa.kernel(u'问下厕所在哪'))