#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This class is very simple and is stateless
"""
import requests
import random

class QAKernel:

    qa_url = 'http://localhost:11403/solr/qa/select?wt=json&q=question:'
    def __init__(self):
        print('initilizing qa kernel...')

    def kernel(self, q):
        r = self._request_solr(q)
        answer = self._extract_answer(r)
        return answer

    def _extract_answer(self, r, random_range=1):
        num = self._num_answer(r)
        if num > 0:
            x = random.randint(0, min(random_range - 1, num - 1))
            response = self._get_response(r, x)
            return response
        else:
            return "我没听懂！"

    def _request_solr(self, q):
        url = self.qa_url + q
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