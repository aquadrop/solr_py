#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This class is very simple and is stateless
<delete><query>*:*</query></delete>
"""
import requests
import random

class IKernel:

    i_url = 'http://localhost:11403/solr/interactive/select?wt=json&q=g:%s OR exact_g:%s^4'
    simple_context_i_url = 'http://localhost:11403/solr/interactive/select?wt=json&q=g:%s^10 OR exact_g:%s^20 OR last_g:%s^2 OR exact_last_g:%s^8'
    def __init__(self):
        print('initilizing interactive kernel...')
        self.last_g = None

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
        if not self.last_g:
            url = self.i_url % (q, q)
            self.last_g = q
        else:
            url = self.simple_context_i_url % (q, q, self.last_g, self.last_g)
            self.last_g = q
        print('debug:interactive_url', url)
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