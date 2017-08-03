#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This class is very simple and is stateless
<delete><query>*:*</query></delete>
"""
import requests
import random
import numpy as np

from client.query_util import QueryUtils
from qa_kernel import QAKernel
import cn_util

class GreetingKernel(QAKernel):

    i_url = 'http://localhost:11403/solr/interactive/select?wt=json&q=g:(%s) OR exact_g:(%s)^4'
    simple_context_i_url = 'http://localhost:11403/solr/interactive/select?wt=json&q=g:(%s)^10 OR exact_g:(%s)^20 OR last_g:(%s)^2 OR exact_last_g:(%s)^8'

    # null_anwer = ['我没听懂您的意思', '我好像不明白...[晕][晕][晕]', '[晕][晕][晕]您能再说一遍吗?我刚刚没听清']
    null_answer = ['null']

    def __init__(self):
        print('attaching greeting kernel...')
        self.last_g = None
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
                return np.random.choice(self.null_anwer, 1, p=[0.5, 0.5])[0]
        except:
            return np.random.choice(self.null_anwer, 1)[0]

    def _request_solr(self, q):
        tokenized, exact_q = self.purify_q(q)
        if not self.last_g:
            url = self.i_url % (tokenized, exact_q)
            self.last_g = q
        else:
            last_tkz, last_exact_q = self.purify_q(self.last_g)
            url = self.simple_context_i_url % (tokenized, exact_q, last_tkz, last_exact_q)
            self.last_g = q
        cn_util.print_cn('debug:interactive_url:' + url)
        r = requests.get(url)
        return r

    def clear_state(self):
        self.last_g = None

    def _num_answer(self, r):
        return int(r.json()["response"]["numFound"])

    def _get_response(self, r, i = 0):
        try:
            a = r.json()["response"]["docs"][i]['b']
            x = random.randint(0, len(a) - 1)
            return a[x].encode('utf8')
        except:
            return None

    def purify_q(self, q):
        q = self.qu.remove_cn_punct(q)
        pos_q = self.qu.corenlp_cut(q, remove_tags=["CD", "PN", "VA", "AD", "VC","SP"])
        return ''.join(pos_q), q