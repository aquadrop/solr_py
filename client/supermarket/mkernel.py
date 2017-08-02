#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import json
import re
import cPickle as pickle
import jieba
import _uniout
import random

import cn_util
import cn2arab

from client.graph import Graph
from client.node import Node
from client.query_util import QueryUtils
from classifier import SeqClassifier


class MKernel:

    def __init__(self, graph_path, clf_path):
        # self.tokenizer = CoreNLP()
        self.graph = None
        self.gbdt = None
        self.state_cleared = True
        self._load_graph(graph_path)
        self._load_clf(clf_path)

        self.qu = QueryUtils()

    last_slots = None

    dialogue_url = "http://localhost:11403/solr/supermarket_dialogue/select?defType=edismax&indent=on&wt=json&rows=1"
    qa_url = "http://localhost:11403/solr/supermarket_qa/select?defType=edismax&indent=on&wt=json&rows=1"
    greeting_url = "http://localhost:11403/solr/supermarket_greeting/select?defType=edismax&indent=on&wt=json&rows=1"
    # tokenizer_url = "http://localhost:5000/pos?q="

    def kernel(self, query):
        return self.r_walk(query=query)

    def clear_state(self):
        print 'state cleared'
        self.state_cleared = True
        self.last_slots = None

    def _load_clf(self, path):
        print('loading gbdt classifier...')
        with open(path, 'rb') as f:
            self.gbdt = pickle.load(f)

    def _load_graph(self, path):
        print('loading logic graph...')
        with open(path, "rb") as input_file:
            self.graph = pickle.load(input_file)

    def num_answer(self, r):
        return int(r.json()["response"]["numFound"])

    def travel_with_clf(self, single_slot, query, gbdt_recursion=True):
        node = self.graph.get_global_node(single_slot)
        slots_list = self.gbdt.predict(parent_slot=single_slot, )


    def r_walk_with_pointer_with_clf(self, query):
        r = None
        response = None
        query = self.qu.remove_cn_punct(query)
        if self.state_cleared:
            ## set state_cleared to False
            self.state_cleared = False
            url = self.base_url + "&q=question:" + query
            print('extact_try_url', url)
            r = requests.get(url)

            if self.num_answer(r) > 0:

                return_slot = self.last_slots = self.get_intention(r)
                self.should_clear_state(self.last_slots)
                return_response = self.get_response(r)
                print('clear exact_', return_slot, self.get_response(r))
                return return_slot, return_response
            else:
                parent_slot = self.graph.slot
                moved, slots = self.travel_with_clf(parent_slot, query)
                ## The intention is moved away from the original one
                if moved:
                    return self.search(parent_slots=self.last_slots, current_slots=slots)
                else:
                    return None, self.trick(query)

        else:
            node = self.graph.all_nodes[self.single_parent_slot()]
            moved, slots = self.travel_with_clf(node, query)
            if moved:
                return self.search(parent_slots=self.last_slots, current_slots=slots)
            else:
                return None, self.trick(query)
    
    def single_parent_slot(self):
        return self.single_slot(self.last_slots)
    
    def single_slot(self, slots):
        return '#'.join(slots)
    
    def search(self, parent_slots, current_slots):
        moved_parent_slot = ''
        if self.last_slots:
            moved_parent_slot = self.single_slot(parent_slots)
        current_slot = self.single_slot(current_slots)
        if moved_parent_slot:
            url = self.base_url + "&fq=intention:(%s)&fq=super_intention:(%s)" % (current_slot, moved_parent_slot)
        else:
            url = self.base_url + "&fq=intention:(%s)" % current_slot
        print("gbdt_result_url", url)
        r = requests.get(url)
        if self.num_answer(r) > 0:
            self.last_slots = current_slots
            self.should_clear_state(current_slots)
            return current_slots, self.get_response(r)
    
    def trick(self, query):
        # ## do trick
        self.clear_state()
        # url = self.concat_solr_request(query=query, base_url=self.trick_url)
        # r = requests.get(url)
        # if self.num_answer(r) > 0:
        #     x = random.randint(0, min(9, self.num_answer(r) - 1))
        #     response = self.get_response(r, x)
        #     return response
        # else:
        #     return "我没听懂！"
        return "我不知道"

    def should_clear_state(self, slot):
        try:
            node = self.graph.get_global_node(slot)
            if node.is_leaf():
                self.clear_state()
        except:
            self.clear_state()

    def get_response(self, r, i=0):
        try:
            a = r.json()["response"]["docs"]
            return r.json()["response"]["docs"][i]["answer"][0].encode('utf8')
        except:
            return None

    def get_last_intention(self, r):
        try:
            return r.json()["response"]["docs"][0]["last_intention"][0].encode('utf8')
        except:
            return None

    def get_next_intention(self, r):
        try:
            return r.json()["response"]["docs"][0]["next_intention"][0].encode('utf8')
        except:
            return None

    def get_intention(self, r):
        try:
            return r.json()["response"]["docs"][0]["intention"][0].encode('utf8')
        except:
            return None

    def concat_solr_request(self, query, base_url, last_intention="", str_slot="", num_slot=0):
        safe_cat = False
        url = base_url + "&q="
        if query:
            url = url + "question:" + query
            safe_cat = True
        if last_intention:
            if safe_cat:
                url = url + "%20AND%20last_intention:" + last_intention
            else:
                url = url + "last_intention:" + last_intention
                safe_cat = True
        if str_slot:
            if safe_cat:
                if last_intention:
                    url = url + "%20OR%20intention:" + str_slot
                else:
                    url = url + "%20OR%20intention:" + str_slot
            else:
                url = url + "intention:" + str_slot
                safe_cat = True
        if num_slot > 0:
            url = url + "&fq=amount_upper:[" + str(
                num_slot) + " TO *]" + "&fq=amount_lower:[0 TO " + str(num_slot) + "]"

        print url

        return url

if __name__ == "__main__":
    K = MKernel("../model/graph_v7.pkl", "../model/seq_clf_v7.pkl")
    while 1:
        ipt = raw_input()
        # chinese comma
        tt = ipt.split("，")
        ##response = K.kernel(ipt)
        s = None
        if len(tt) >= 2:
            q = tt[0]
            s = tt[1]
        q = tt[0]
        print q, s
        if s:
            next_slot, response = K.r_walk_with_pointer_with_clf(
                q, s.encode('utf8'))
        else:
            next_slot, response = K.r_walk_with_pointer_with_clf(q)
        print(str(response))
