#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np

class SolrUtils:
    def __init__(self):
        print('attaching solr utils')

    @staticmethod
    def num_answer(r):
        return int(r.json()["response"]["numFound"])

    @staticmethod
    def get_response(r, i=0):
        try:
            a = r.json()["response"]["docs"][i]["answer"]
            x = random.randint(0, len(a) - 1)
            return a[x].encode('utf8')
        except:
            return None

    @staticmethod
    def get_parent_intention(r):
        try:
            return r.json()["response"]["docs"][0]["super_intention"][0].encode('utf8')
        except:
            return None

    @staticmethod
    def get_intention(r):
        try:
            return r.json()["response"]["docs"][0]["intention"]
        except:
            return None

    @staticmethod
    def get_dynamic_response(r, key, random_hit=False, random_field=True, keep_array=False):
        try:
            num = np.minimum(SolrUtils.num_answer(r), 10)
            if random_hit:
                hit = np.random.randint(0, num)
            else:
                hit = 0
            a = r.json()["response"]["docs"][hit][key]
            if keep_array:
                return a
            else:
                if random_field:
                    rr = np.random.choice(a, 1)[0]
                else:
                    rr = ','.join(a)
            return rr.encode('utf8')
        except:
            return None