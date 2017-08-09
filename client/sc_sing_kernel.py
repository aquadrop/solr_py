#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This class is very simple and is stateless
"""
import requests
import random
import jieba
import re

import numpy as np

from query_util import QueryUtils
import cn_util

class SimpleSingKernel:
    null_anwer = ['额...我只会唱周杰伦的星睛,轨迹,七里香']
    # null_answer = ['null']

    def __init__(self):
        print('attaching sing kernel...')
        ## http://localhost:11403/solr/sc_qa/select?fq=entity:%E5%8E%95%E6%89%80&indent=on&q=*:*&wt=json
        self.qa_url = 'http://localhost:11403/solr/sc_music_kb/select?q.op=OR&wt=json&q=%s'
        self.qu = QueryUtils()

    def kernel(self, q):
        # response = self.regex_plugin(q)
        # if response:
        #     return response
        try:
            r = self._request_solr(q)
            answer = self._extract_answer(r)
            return answer
        except Exception,e:
            return 'mainframe unable to reply since sing kernel damaged...'


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
        url = self.qa_url % q
        # print('qa_debug:', url)
        # cn_util.print_cn(url)
        r = requests.get(url)
        return r

    def _num_answer(self, r):
        return int(r.json()["response"]["numFound"])

    def _get_response(self, r, i = 0):
        try:
            a = r.json()["response"]["docs"][i]['script']
            rr = np.random.choice(a, 1)[0]
            x = random.randint(0, min(0, len(a) - 1))
            return rr.encode('utf8')
        except:
            return np.random.choice(self.null_anwer, 1)[0]

    def purify_q(self, q):
        q = self.qu.remove_cn_punct(q)
        pos_q = self.qu.corenlp_cut(q, remove_tags=["CD", "VA", "AD", "VC"])
        return ''.join(pos_q), q

if __name__ == '__main__':
    qa = SimpleSingKernel()
    print(qa.kernel(u'唱首周杰伦的歌'))