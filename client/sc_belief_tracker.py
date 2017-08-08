#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import cPickle as pickle

from node import Node
from query_util import QueryUtils
from sc_belief_clf import Multilabel_Clf
from solr_utils import SolrUtils
import cn_util
from sc_belief_graph import BeliefGraph

class BeliefTracker:
    def __init__(self, graph_path, clf_path):
        self.gbdt = None
        self.state_cleared = True
        self._load_graph(graph_path)
        self._load_clf(clf_path)
        self.search_graph = BeliefGraph()
        ## keep track of remaining slots, the old slots has lower score index, if index = -1, remove that slot
        self.remaining_slots = {}
        self.score_stairs = [2,4,6,8,10]
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
        except Exception, e:
            print('failed to attach main kernel...detaching...', e.message)

    def _load_graph(self, path):
        try:
            print('attaching logic graph...100%')
            with open(path, "rb") as input_file:
                self.belief_graph = pickle.load(input_file)
                # print(self.graph.go('购物', Node.REGEX))
        except:
            print('failed to attach logic graph...detaching...')

    def travel_with_clf(self, query):
        filtered_slots_list = []
        if self.gbdt:
            slots_list, probs = self.gbdt.predict(input_=query)
            for i, prob in enumerate(probs):
                if prob >= 0.7:
                    cn_util(slots_list[i])
                    # print('classifying...', prob)
                    filtered_slots_list.append(slots_list[i])

            filtered_slots_list = set(filtered_slots_list)
            if len(filtered_slots_list) == 0:
                return False, []
        else:
            raise Exception('malfunctioning, main kernel must be attached!')

        ## build belief graph
        self.update_remaining_slots(expire=True)
        self.update_belief_graph(search_parent_node=self.search_graph, slots_list=filtered_slots_list)

    def update_belief_graph(self, search_parent_node, slots_list, slots_marker=None):
        if not slots_marker:
            slots_marker = [0] * len(slots_list)
        slots_list = list(set(slots_list))
        search_parent_slot = search_parent_node.slot
        defined_parent_node = self.belief_graph.get_global_node(slot=search_parent_slot)
        for i, slot in enumerate(slots_list):
            if slots_marker[i] == 1:
                continue
            # check if child node
            if defined_parent_node.has_child(slot, Node.KEY):
                slots_marker[i] == 1
                # check search_node type
                if search_parent_node.has_child(slot, value_type=Node.KEY):
                    self.update_remaining_slots(slot)
                    continue
                type_ = self.belief_graph.slot_identities[slot]
                for key, value in search_parent_node.all_children(value_type=Node.KEY).iteritems():
                    type2_ = self.belief_graph.slot_identities[key]
                    ## replace the old one with new one
                    if type2_ == type_:
                        search_parent_node.remove_node(key=key, value_type=Node.KEY)
                ## new slot added
                new_node = Node(slot=slot)
                search_parent_node.add_node(node=new_node, value_type=Node.KEY, values=[slot])
                self.update_remaining_slots(slot)
                defined_new_node = self.belief_graph.get_global_node(slot)
                if not defined_new_node.is_leaf(Node.KEY):
                    self.update_belief_graph(new_node, slots_list, slots_marker)

    def update_remaining_slots(self, slot=None, expire=False):
        if expire:
            for remaining_slot, index in self.remaining_slots.iteritems():
                self.remaining_slots[remaining_slot] = index - 1
            self.remaining_slots = {k: v for k, v in self.remaining_slots.iteritems() if v >= 0}
        if slot:
            self.remaining_slots[slot] = len(self.score_stairs) - 1

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
        return self.single_slot(self.last_slots, split=split)

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
    bt = BeliefTracker("../model/sc/belief_graph.pkl", '../model/sc/belief_clf.pkl')
    while 1:
        ipt = raw_input()
        # chinese comma
        bt.travel_with_clf(ipt)
        cn_util.print_cn(bt.remaining_slots)
