#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This class is very simple and is stateless
<delete><query>*:*</query></delete>
"""
import requests
import random
import numpy as np

from query_util import QueryUtils
from sc_qa_kernel import QAKernel
import cn_util

class GreetingKernel(QAKernel):
    # simple_context_i_url = 'http://localhost:11403/solr/interactive/select?wt=json&q=g:(%s)^10 OR exact_g:(%s)^20 OR last_g:(%s)^2 OR exact_last_g:(%s)^8'

    # null_anwer = ['我没听懂您的意思', '我好像不明白...[晕][晕][晕]', '[晕][晕][晕]您能再说一遍吗?我刚刚没听清']
    null_answer = ['null']

    def __init__(self):
        print('attaching greeting kernel...')
        self.last_g = None
        self.qu = QueryUtils()
        self.greeting_url = 'http://localhost:11403/solr/sc_greeting/select?q.op=OR&wt=json&q=question:(%s)'
        self.exact_greeting_url = 'http://localhost:11403/solr/sc_greeting/select?wt=json&q=exact_question:(%s)'

    def exact_match(self, q, random_range=1):
        url = self.exact_greeting_url % q
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

    def _request_solr(self, q):
        tokenized, exact_q = self.purify_q(q)
        url = self.greeting_url % tokenized.decode('utf-8')
        # print('qa_debug:', url)
        r = requests.get(url)
        return r