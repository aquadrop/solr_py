#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This class is very simple and is stateless
"""
import requests
import random

import numpy as np

from query_util import QueryUtils

class QAKernel:

    qa_url = 'http://localhost:11403/solr/qa/select?wt=json&q=question:(%s)'

    # null_anwer = ['这个我不知道,您可以谷歌或百度', '我知识有限,这个我不知道怎么回答...[晕][晕][晕]']
    null_answer = ['null']

    def __init__(self):
        print('initilizing qa kernel...')
        self.qu = QueryUtils()

    def kernel(self, q):
        r = self._request_solr(q)
        answer = self._extract_answer(r)
        return answer

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
        tokenized, exact_q = self.purify_q(q)
        url = self.qa_url % tokenized.decode('utf-8')
        print('qa_debug:', url)
        r = requests.get(url)
        return r

    def _num_answer(self, r):
        return int(r.json()["response"]["numFound"])

    def _get_response(self, r, i = 0):
        try:
            a = r.json()["response"]["docs"]
            return a[i]["answer"][0].encode('utf8')
        except:
            return None

    def purify_q(self, q):
        q = self.qu.remove_cn_punct(q)
        pos_q = self.qu.corenlp_cut(q, remove_tags=["CD", "VA", "AD", "VC"])
        return ''.join(pos_q), q