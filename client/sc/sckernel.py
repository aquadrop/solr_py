#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import cPickle as pickle

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

    guide_url = "http://localhost:11403/solr/sc_guide/select?defType=edismax&indent=on&wt=json&q=*:*"
    qa_url = "http://localhost:11403/solr/sc_qa/select?defType=edismax&indent=on&wt=json&q=*:*"
    greeting_url = "http://localhost:11403/solr/sc_greeting/select?defType=edismax&indent=on&wt=json&q=*:*"

    # tokenizer_url = "http://localhost:5000/pos?q="

    def kernel(self, query):
        return self.r_walk_with_pointer_with_clf(query=query)

    def clear_state(self):
        print 'state cleared'
        self.state_cleared = True
        self.last_slots = None

    def _load_clf(self, path):
        try:
            print('attaching gbdt classifier...100%')
            self.gbdt = Multilabel_Clf.load(path)
        except Exception, e:
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
            slots_list, probs = self.gbdt.predict(parent_slot=single_slot, input_=query)
            for i, prob in enumerate(probs):
                if prob >= 0.9:
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
        return moved, rec_slots_list

    def r_walk_with_pointer_with_clf(self, query):
        r = None
        response = None
        query = self.qu.remove_cn_punct(query)
        if self.state_cleared:
            self.flag_state()
            ## set state_cleared to False
            parent_slot = self.graph.slot
            moved, slots = self.travel_with_clf(parent_slot, query)
            ## The intention is moved away from the original one
            if moved:
                return self.search(parent_slots=self.last_slots, current_slots=slots)
            else:
                return 'unclear', self.trick(query)

        else:
            moved, slots = self.travel_with_clf(self.single_parent_slot(), query)
            if moved:
                return self.search(parent_slots=self.last_slots, current_slots=slots)
            else:
                return 'unclear', self.trick(query)

    def single_parent_slot(self, split=' OR '):
        return self.single_slot(self.last_slots, split=split)

    def single_slot(self, slots, split=' OR '):
        return split.join(slots)

    def flag_state(self):
        self.state_cleared = False

    def search(self, parent_slots, current_slots):
        try:
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
        except:
            return 'unclear', self.travel_with_clf(None)

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
