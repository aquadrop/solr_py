#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import traceback
import cPickle as pickle

import numpy as np

from node import Node
from query_util import QueryUtils
from sc_belief_clf import Multilabel_Clf
from solr_utils import SolrUtils
import cn_util
from sc_belief_graph import BeliefGraph
from sc_negative_clf import Negative_Clf
from sc_simple_qa_kernel import SimpleQAKernel
from sc_qa_kernel import QAKernel

class BeliefTracker:
    ## static
    static_gbdt = None
    static_belief_graph = None
    static_qa_clf = None

    def __init__(self, graph_path, clf_path):
        self.gbdt = None
        self.state_cleared = True
        self._load_graph(graph_path)
        self._load_clf(clf_path)
        self.search_graph = BeliefGraph()
        ## keep track of remaining slots, the old slots has lower score index, if index = -1, remove that slot
        self.remaining_slots = {}
        self.negative_slots = {}
        self.score_stairs = [1, 4, 16, 64, 256]
        self.negative = False
        self.negative_clf=Negative_Clf()
        self.simple = SimpleQAKernel()
        self.machine_state = None

    last_slots = None

    guide_url = "http://localhost:11403/solr/sc_sale_gen/select?defType=edismax&indent=on&wt=json"
    # tokenizer_url = "http://localhost:5000/pos?q="

    def kernel(self, query):
        query = QueryUtils.static_simple_remove_punct(query)
        next_scene, inside_intentions, response = self.r_walk_with_pointer_with_clf(query=query)
        return next_scene, inside_intentions, response

    def clear_state(self):
        self.search_graph = BeliefGraph()
        self.remaining_slots = {}
        self.negative_slots = {}
        self.negative = False

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
        if not BeliefTracker.static_belief_graph:
            try:
                print('attaching logic graph...100%')
                with open(path, "rb") as input_file:
                    self.belief_graph = pickle.load(input_file)
                    BeliefTracker.static_belief_graph = self.belief_graph
                    # print(self.graph.go('购物', Node.REGEX))
            except:
                print('failed to attach logic graph...detaching...')
        else:
            print('skipping attaching logic graph...')
            self.belief_graph = BeliefTracker.static_belief_graph

    def travel_with_clf(self, query):
        filtered_slots_list = []
        if self.gbdt:
            flipped, self.negative = self.negative_clf.predict(input_=query)
            slots_list, probs = self.gbdt.predict(input_=flipped)
            if not slots_list or len(slots_list) == 0:
                print('strange empty predcition')
            # print self.negative
            for i, prob in enumerate(probs):
                if prob >= 0.7:
                    filtered_slots_list.append(slots_list[i])
                else:
                    cn_util.print_cn("droping slot:", slots_list[i], str(prob))

            filtered_slots_list = set(filtered_slots_list)
            if len(filtered_slots_list) == 0:
                print('valid empty predcition')
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
                        if key in self.remaining_slots:
                            del self.remaining_slots[key]
                        if key in self.negative_slots:
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
                if remaining_slot in ['随便']:
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

    def remove_slots(self, key):
        new_remaining_slots = {}
        for remaining_slot, index in self.remaining_slots.iteritems():
            if remaining_slot == key:
                continue
            node = self.belief_graph.get_global_node(remaining_slot)
            if node.has_ancester(key):
                continue
            new_remaining_slots[remaining_slot] = self.remaining_slots[remaining_slot]
        self.remaining_slots = new_remaining_slots

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
                    slot = self.sibling(slot=slot, maximum_num=1)[0]
            elif size == 1:
                if self.negative_slots[slot]:
                    slot = self.sibling(slot=slot, maximum_num=1)[0]
            intention = slot + '^' + str(float(score) * float(importance))
            intentions.append(intention)
        return intentions, ' OR '.join(intentions)

    def contain_negative(self, intentions):
        for intention in intentions:
            if "-" in intention:
                return True

        return False

    def sibling(self, slot, maximum_num):
        black_list = ['facility', 'entertainment']
        node = self.belief_graph.get_global_node(slot)
        sibling = node.sibling_names(value_type=Node.KEY)
        ## must be of same identities
        identity = self.belief_graph.slot_identities[slot.encode('utf-8')]
        cls_sibling = []
        for s in sibling:
            try:
                if s in black_list:
                    continue
                if self.belief_graph.slot_identities[s.encode('utf-8')] == identity:
                    cls_sibling.append(s)
            except:
                pass
        maximum_num = np.minimum(maximum_num, len(cls_sibling))
        return np.random.choice(a=cls_sibling, replace=False, size=maximum_num)

    def search(self):
        try:
            intentions, fq = self.compose()
            if len(self.remaining_slots) == 0:
                return 'base', 'empty', None
            if 'facility' in self.remaining_slots or 'entertainment' in self.remaining_slots:
                qa_intentions = ','.join(self.remaining_slots)
                self.remove_slots('facility')
                self.remove_slots('entertainment')
                _, response = self.simple.kernel(qa_intentions)
                return None, ','.join(intentions), response
            url = self.guide_url + "&q=intention:(%s)" % fq
            cn_util.print_cn("gbdt_result_url", url)
            r = requests.get(url)
            if SolrUtils.num_answer(r) > 0:
                response = self._get_response(r, 'answer', random_hit=self.contain_negative(intentions), random_field=True, keep_array=False)
                labels = self._get_response(r, 'intention', random_field=True, keep_array=True)
                remove_labels = [u"购物", u"吃饭",u"随便"]
                for rl in remove_labels:
                    if rl in labels:
                        labels.remove(rl)
                response = self.generate_response(response, labels)
                return None, ','.join(intentions), response
        except:
            traceback.print_exc()
            return 'base', 'error', '我好像不知道哦, 问问咨询台呢'

    def generate_response(self, response, labels):
        graph_url = 'http://localhost:11403/solr/graph/select?wt=json&q=%s'
        if '<s>' in response:
            condition = []
            for label in labels:
                try:
                    string = 'label:%s' % (label + '^' + str(self.belief_graph.slot_importances[label.encode('utf-8')]))
                except:
                    string = 'label:%s' % label
                condition.append(string)
            condition = '%20OR%20'.join(condition)
            url = graph_url % condition
            cn_util.print_cn(url)
            r = requests.get(url)
            if SolrUtils.num_answer(r) > 0:
                name = self._get_response(r=r, key='name', random_hit=True, random_field=True)
                location = self._get_response(r=r, key='location', random_hit=True, random_field=True)
                new_response = response.replace('<s>', name).replace('<l>', location)
                return new_response
            else:
                return '没有找到相关商家哦.您的需求有点特别哦.或者不在知识库范围内...'
        else:
            return response

    def _get_response(self, r, key, random_hit=False, random_field=True, keep_array=False):
        try:
            num = np.minimum(SolrUtils.num_answer(r), 3)
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
    ipts = ["买鲜花"]
    for ipt in ipts:
        # ipt = raw_input()
        # chinese comma
        # bt.travel_with_clf(ipt)
        cn_util.print_cn(",".join(bt.kernel(ipt)))
        # cn_util.print_cn(bt.compose()[0])
