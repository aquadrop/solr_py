#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

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