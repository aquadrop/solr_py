#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import cPickle as pickle

from node import Node
from query_util import QueryUtils
from sc_multilabel_clf import Multilabel_Clf
from solr_utils import SolrUtils

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

    guide_url = "http://localhost:11403/solr/sc_sale/select?defType=edismax&indent=on&wt=json&q=*:*"
    # tokenizer_url = "http://localhost:5000/pos?q="

    def kernel(self, query):
        query = self.qu.remove_cn_punct(query)
        return self.r_walk_with_pointer_with_clf(query=query)

    def clear_state(self):
        print 'state cleared'
        self.state_cleared = True
        # self.last_slots = None

    def _load_clf(self, path):
        try:
            print('attaching gbdt classifier...100%')
            with open(path, "rb") as input_file:
                self.gbdt = pickle.load(input_file)
            # self.gbdt = Multilabel_Clf.load(path)
        except Exception,e:
            print('failed to attach gbdt classifier...detaching...', e.message)

    def _load_graph(self, path):
        try:
            print('attaching logic graph...100%')
            with open(path, "rb") as input_file:
                self.graph = pickle.load(input_file)
                # print(self.graph.go('购物', Node.REGEX))
        except:
            print('failed to attach logic graph...detaching...')

    ## splits slots_list into parent_slot and vertical ones
    def tree_filter(self, slots_list):
        return slots_list[0], slots_list[1:-1]

    def travel_with_clf(self, single_slot, query):
        filtered_slots_list = []
        moved = True
        if self.gbdt:
            if not single_slot:
                single_slot = 'ROOT'
            if single_slot in self.graph.all_nodes:
                node = self.graph.get_global_node(single_slot)
                if node.is_leaf(value_type=Node.REGEX):
                    return False, []
            else:
                return False, []
            slots_list, probs = self.gbdt.predict(parent_slot=single_slot, input_=query)
            for i, prob in enumerate(probs):
                if prob >= 0.7:
                    # print('classifying...', slots_list[i], prob)
                    filtered_slots_list.append(slots_list[i])

            filtered_slots_list = set(filtered_slots_list)
            if len(filtered_slots_list) == 0:
                return False, []
        else:
            node = self.graph.get_global_node(single_slot)
            children = node.go(q=query, value_type=Node.REGEX)
            for c in children:
                filtered_slots_list.append(c.slot)

            if len(filtered_slots_list) == 0:
                return False, []

        rec_slots_list = []
        rec_slots_list.extend(filtered_slots_list)
        for slot in filtered_slots_list:
            next_node = self.graph.get_global_node(slot)
            if not next_node.is_leaf(Node.REGEX):
                next_node_single_slot = next_node.slot
                next_moved, next_slots_list = self.travel_with_clf(next_node_single_slot, query)
                if next_moved:
                    ## replace the parent slot
                    rec_slots_list.remove(slot)
                    rec_slots_list.extend(next_slots_list)

        rec_slots_list = set(rec_slots_list)
        return moved, list(rec_slots_list)

    def r_walk_with_pointer_with_clf(self, query, parent_slot=None):
        ## priority to top-bottom
        if self.state_cleared:
            self.flag_state()
            ## set state_cleared to False
            # parent_slot = self.graph.slot
            parent_slot = 'ROOT'
        else:
            if not parent_slot:
                parent_slot = self.single_last_slot(split='|')
        moved, slots = self.travel_with_clf(parent_slot, query)
        ## The intention is moved away from the original one
        if moved:
            return self.search(current_slots=slots)
        else:
            last_slot = self.last_slots[0]
            last_node = self.graph.get_global_node(last_slot)
            if last_node.is_root():
                return 'unclear', self.trick(query)
            else:
                parent_slot = last_node.parent_node.slot
                return self.r_walk_with_pointer_with_clf(query, parent_slot)
    
    def single_last_slot(self, split=' OR '):
        return self.single_slot(self.last_slots, split=split    )
    
    def single_slot(self, slots, split=' OR '):
        return split.join(slots)

    def flag_state(self):
        self.state_cleared = False

    def search(self, current_slots):
        try:
            current_slot = self.single_slot(current_slots)
            url = self.guide_url + "&fq=intention:(%s)" % current_slot
            # print("gbdt_result_url", url)
            r = requests.get(url)
            if SolrUtils.num_answer(r) > 0:
                self.last_slots = current_slots
                self.should_clear_state(current_slots)
                return current_slots, SolrUtils.get_response(r)
        except:
            return 'unclear', 'out of domain knowledge'
    
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
        return "out of domain knowledge"

    def should_clear_state(self, multi_slots):
        try:
            single_slot = self.single_slot(multi_slots)
            node = self.graph.get_global_node(single_slot)
            if node.is_leaf(Node.REGEX):
                self.clear_state()
        except:
            self.clear_state()


if __name__ == "__main__":
    SC = SCKernel("../../model/sc_graph_v7.pkl", '../../model/sc/multilabel_clf.pkl')
    while 1:
        ipt = raw_input()
        # chinese comma
        next_slot, response = SC.r_walk_with_pointer_with_clf(ipt)
        print(str(response))
