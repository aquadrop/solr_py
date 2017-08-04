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
from client.sc.qa_kernel import QAKernel
import cn_util

class GreetingKernel(QAKernel):
    # simple_context_i_url = 'http://localhost:11403/solr/interactive/select?wt=json&q=g:(%s)^10 OR exact_g:(%s)^20 OR last_g:(%s)^2 OR exact_last_g:(%s)^8'

    # null_anwer = ['我没听懂您的意思', '我好像不明白...[晕][晕][晕]', '[晕][晕][晕]您能再说一遍吗?我刚刚没听清']
    null_answer = ['null']

    def __init__(self):
        print('attaching greeting kernel...')
        self.last_g = None
        self.qu = QueryUtils()
        self.greeting_url = 'http://localhost:11403/solr/sc_greeting/select?wt=json&q=question:(%s)'

    def _request_solr(self, q):
        tokenized, exact_q = self.purify_q(q)
        url = self.greeting_url % tokenized.decode('utf-8')
        print('qa_debug:', url)
        r = requests.get(url)
        return r