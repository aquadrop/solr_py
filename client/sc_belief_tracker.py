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
from sc_negative_clf import Negative_Clf


class BeliefTracker:
    ## static
    static_gbdt = None

    def __init__(self, graph_path, clf_path):
        self.gbdt = None
        self.state_cleared = True
        self._load_graph(graph_path)
        self._load_clf(clf_path)
        self.search_graph = BeliefGraph()
        ## keep track of remaining slots, the old slots has lower score index, if index = -1, remove that slot
        self.remaining_slots = {}
        self.negative_slots = {}
        self.score_stairs = [1, 5, 10, 20, 40]
        self.qu = QueryUtils()
        self.negative_clf=Negative_Clf()

    last_slots = None

    guide_url = "http://localhost:11403/solr/sc_sale_adv/select?defType=edismax&indent=on&wt=json"

    # tokenizer_url = "http://localhost:5000/pos?q="

    def kernel(self, query):
        query = self.qu.remove_cn_punct(query)
        return self.r_walk_with_pointer_with_clf(query=query)

    def clear_state(self):
        self.search_graph = BeliefGraph()

    def _load_clf(self, path):
        if not BeliefTracker.static_gbdt:
            try:
                print('attaching gbdt classifier...100%')
                with open(path, "rb") as input_file:
                    self.gbdt = pickle.load(input_file)
                    BeliefTracker.static_gbdt = self.gbdt
                    # self.gbdt = Multilabel_Clf.load(path)
            except Exception, e:
                print('failed to attach main kernel...detaching...', e.message)
        else:
            print('skipping attaching gbdt classifier as already attached...')
            self.gbdt = BeliefTracker.static_gbdt

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
            self.negative = self.negative_clf.predict(input_=query)
            # print self.negative
            for i, prob in enumerate(probs):
                if prob >= 0.7:
                    filtered_slots_list.append(slots_list[i])

            filtered_slots_list = set(filtered_slots_list)
            if len(filtered_slots_list) == 0:
                return False, []
        else:
            raise Exception('malfunctioning, main kernel must be attached!')

        ## build belief graph
        self.update_remaining_slots(expire=True)
        filtered_slots_list = self.inter_fix(filtered_slots_list)
        self.should_expire_all_slots(filtered_slots_list)
        self.update_belief_graph(search_parent_node=self.search_graph, slots_list=filtered_slots_list)

    def should_expire_all_slots(self, slots_list):
        slots_list = list(slots_list)
        if len(slots_list) == 1:
            slot = slots_list[0]
            if self.belief_graph.has_child(slot, Node.KEY) \
                    and self.belief_graph.slot_identities[slot] == 'intention':
                self.remaining_slots.clear()
                self.negative_slots.clear()
                self.search_graph = BeliefGraph()

    ## fill slots when incomplete
    ## silly fix
    def inter_fix(self, slots_list):
        ## check broken
        try:
            broken = True
            for slot in slots_list:
                if self.belief_graph.has_child(key=slot, value_type=Node.KEY):
                    broken = False
                    break
            if not broken:
                return slots_list

            max_go_up = 10
            filled_slots_list = list(set(slots_list))
            for slot in slots_list:
                current_slot = slot
                for i in xrange(max_go_up):
                    node = self.belief_graph.get_global_node(current_slot)
                    parent_node = node.parent_node
                    current_slot = parent_node.slot
                    if not parent_node.is_root():
                        filled_slots_list.append(parent_node.slot)
                    else:
                        break

            return list(set(filled_slots_list))
        except:
            return slots_list

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
                # set 0 if negative
                slots_marker[i] == 1
                # check search_node type
                if search_parent_node.has_child(slot, value_type=Node.KEY):
                    # self.update_remaining_slots(slot)
                    new_node = search_parent_node.get_child(slot, Node.KEY)
                else:
                    type_ = self.belief_graph.slot_identities[slot]
                    remove_keys = []
                    for key, value in search_parent_node.all_children(value_type=Node.KEY).iteritems():
                        type2_ = self.belief_graph.slot_identities[key]
                        ## replace the old one with new one
                        if type2_ == type_:
                            ## do this recursively, delete all associated children
                            remove_keys.append(key)
                            sub_children_names = value.all_children_names_recursive(Node.KEY)
                            remove_keys.extend(sub_children_names)

                    for key in remove_keys:
                        del self.remaining_slots[key]
                        del self.negative_slots[key]
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
                ## special clear this slot once
                if remaining_slot == '随便':
                    self.remaining_slots[remaining_slot] = -1
            self.remaining_slots = {k: v for k, v in self.remaining_slots.iteritems() if v >= 0}
        if slot:
            if self.negative:
                self.negative_slots[slot] = True
            else:
                self.negative_slots[slot] = False
            self.remaining_slots[slot] = len(self.score_stairs) - 1

    def r_walk_with_pointer_with_clf(self, query):
        self.travel_with_clf(query)
        return self.search()

    def single_last_slot(self, split=' OR '):
        return self.single_slot(self.last_slots, split=split)

    def single_slot(self, slots, split=' OR '):
        return split.join(slots)

    def flag_state(self):
        self.state_cleared = False

    def compose(self):
        intentions = []
        size = len(self.remaining_slots)
        for slot, i in self.remaining_slots.iteritems():
            node = self.belief_graph.get_global_node(slot)
            score = self.score_stairs[i]
            importance = self.belief_graph.slot_importances[slot]
            if size > 2:
                if self.negative_slots[slot] and node.is_leaf(Node.KEY):
                    slot = '-' + slot
            elif size==2:
                if self.negative_slots[slot] and self.belief_graph.slot_identities[slot] != 'intention':
                    slot = '-' + slot
            elif size == 1:
                if self.negative_slots[slot]:
                    slot = '-' + slot
            intention = slot + '^' + str(float(score) * float(importance))
            intentions.append(intention)
        return intentions, ' OR '.join(intentions)

    def search(self):
        try:
            intentions, fq = self.compose()
            url = self.guide_url + "&q=intention:(%s)" % fq
            # print("gbdt_result_url", url)
            r = requests.get(url)
            if SolrUtils.num_answer(r) > 0:
                return ','.join(intentions), SolrUtils.get_response(r)
        except:
            return 'unclear', '我好像不知道哦, 问问咨询台呢'

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
    ipts = ['我要吃火锅','我不吃饭','不买拖鞋','不吃冰淇淋','吃不起贵的西餐']
    for ipt in ipts:
        # ipt = raw_input()
        # chinese comma
        bt.travel_with_clf(ipt)
        cn_util.print_cn(bt.compose()[0])
