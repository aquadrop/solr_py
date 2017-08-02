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

from client.graph import Graph
from client.node import Node
from client.query_util import QueryUtils
from multilabel_clf import Multilabel_Clf
from client.solr_utils import SolrUtils

class SCKernel:

    def __init__(self, graph_path, clf_path):
        # self.tokenizer = CoreNLP()
        self.graph = None
        self.gbdt = None
        self.state_cleared = True
        self._load_graph(graph_path)
        self._load_clf(clf_path)

        self.qu = QueryUtils()

    last_slots = None

    guide_url = "http://localhost:11403/solr/sc_guide/select?defType=edismax&indent=on&wt=json&rows=1"
    qa_url = "http://localhost:11403/solr/sc_qa/select?defType=edismax&indent=on&wt=json&rows=1"
    greeting_url = "http://localhost:11403/solr/sc_greeting/select?defType=edismax&indent=on&wt=json&rows=1"
    # tokenizer_url = "http://localhost:5000/pos?q="

    def kernel(self, query):
        return self.r_walk(query=query)

    def clear_state(self):
        print 'state cleared'
        self.state_cleared = True
        self.last_slots = None

    def _load_clf(self, path):
        try:
            print('attaching gbdt classifier...100%')
            with open(path, 'rb') as f:
                self.gbdt = pickle.load(f)
        except:
            print('failed to attach gbdt classifier...detaching...')

    def _load_graph(self, path):
        try:
            print('attaching logic graph...100%')
            with open(path, "rb") as input_file:
                self.graph = pickle.load(input_file)
        except:
            print('failed to attach logic graph...detaching...')

    ## splits slots_list into parent_slot and vertical ones
    def tree_filter(self, slots_list):
        return slots_list[0], slots_list[1:-1]

    def travel_with_clf(self, single_slot, query):
        filtered_slots_list = []
        if self.gbdt:
            slots_list, probs = self.gbdt.predict(parent_class=single_slot, question=query)
            moved = True
            for i, prob in enumerate(probs):
                if prob >= 0.9:
                    filtered_slots_list.append(slots_list[i])

            filtered_slots_list = set(filtered_slots_list)
            if len(filtered_slots_list) == 0:
                moved = False
        else:
            node = self.graph.get_global_node(single_slot)
            children = node.go(query, Node.REGEX)
            for c in children:
                filtered_slots_list.append(c.slot)

        rec_slots_list = []
        rec_slots_list.extend(filtered_slots_list)
        for slot in filtered_slots_list:
            next_node = self.graph.get_global_node(slot)
            if not next_node.is_leaf():
                next_node_single_slot = next_node.slot
                next_moved, next_slots_list = self.travel_with_clf(next_node_single_slot, query)
                if next_moved:
                    ## replace the parent slot
                    rec_slots_list.remove(slot)
                    rec_slots_list.extend(next_slots_list)

        rec_slots_list = set(rec_slots_list)
        return moved, rec_slots_list

    def r_walk_with_pointer_with_clf(self, query):
        r = None
        response = None
        query = self.qu.remove_cn_punct(query)
        if self.state_cleared:
            ## set state_cleared to False
            self.state_cleared = False
            url = self.guide_url + "&q=question:" + query
            print('extact_try_url', url)
            r = requests.get(url)

            if SolrUtils.num_answer(r) > 0:
                return_slot = self.last_slots = SolrUtils.get_intention(r)
                self.should_clear_state(self.last_slots)
                return_response = SolrUtils.get_response(r)
                print('clear exact_', return_slot, return_response)
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
        url = self.guide_url + "&fq=intention:(%s)" % current_slot
        print("gbdt_result_url", url)
        r = requests.get(url)
        if SolrUtils.num_answer(r) > 0:
            self.last_slots = current_slots
            self.should_clear_state(current_slots)
            return current_slots, SolrUtils.get_response(r)
    
    def trick(self, query):
        # ## do trick
        self.clear_state()
        # url = self.concat_solr_request(query=query, guide_url=self.trick_url)
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


if __name__ == "__main__":
    SC = SCKernel("../../model/sc_graph_v7.pkl", None)
    while 1:
        ipt = raw_input()
        # chinese comma
        next_slot, response = SC.r_walk_with_pointer_with_clf(ipt)
        print(str(response))
