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

from graph import Graph
from node import Node
from sequence_classifier import SeqClassifier

class GKernel:

    def __init__(self, graph_path, clf_path):
        # self.tokenizer = CoreNLP()
        self.graph = None
        self.gbdt = None
        self.state_cleared = True
        self._load_graph(graph_path)
        self._load_clf(clf_path)

    last_slot = None

    base_url = "http://localhost:11403/solr/qa/select?defType=edismax&indent=on&wt=json&rows=1"
    trick_url = "http://localhost:11403/solr/trick/select?defType=edismax&indent=on&wt=json&rows=10"
    # tokenizer_url = "http://localhost:5000/pos?q="

    def kernel(self, query):
        return self.r_walk(query=query)

    def clear_state(self):
        print 'state cleared'
        self.state_cleared = True
        self.last_slot = None

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

    ## tokens: set
    def travel(self, tokens, node):
        key = None ## word/tag
        next_node = None
        key_found = False
        num_found = False
        for t in tokens:
            try:
                ss = t.split("/")
                word = ss[0]
                tag = ss[1]
                value_types = node.value_types
                if "KEY" in value_types and tag != "CD":
                    try:
                        next_node = node.go(q=word, value_type="KEY")
                        key_found = next_node is not None
                        if key_found:
                            self.last_slot = node.slot
                            key = word + "/" + tag
                            break
                    except:
                        key_found = False
                if "RANGE" in value_types and tag == "CD":
                    try:
                        _word = cn2arab.cn2arab(word)
                        next_node = node.go(q=float(_word), value_type="RANGE")
                        num_found = next_node is not None
                        if num_found:
                            self.last_slot = node.slot
                            key = word + "/" + tag
                            break
                    except:
                        num_found = False
            except:
                return node
        if not key_found and not num_found:
            ## last try solr:

            return node
        else:
            tokens.remove(key)
            print key, next_node.slot, node.slot
            return self.travel(tokens, next_node)

    def r_walk_with_pointer(self, query, given_slot=None):
        r = None
        response = None
        if self.state_cleared:

            url = self.base_url + "&q=exact_question:" + query
            print url
            r = requests.get(url)

            if self.num_answer(r) > 0:
                self.state_cleared = False
                self.last_slot = self.get_intention(r)
                self.should_clear_state(self.last_slot)
                return_slot = self.last_slot
                return_response = self.get_response(r)
                print 'clear exact_', return_slot, self.get_response(r)
                return return_slot, return_response
            else:
                tokenized = self.tokenize(query)
                tokens = tokenized.split(" ")
                print tokens, tokenized
                if not tokenized:
                    ## do trick
                    self.clear_state()
                    # url = self.concat_solr_request(query=query, base_url=self.trick_url)
                    # r = requests.get(url)
                    # response = self.get_response(r)
                    # self.clear_state()
                    # print 'clear non exact_', self.get_intention(r), self.get_response(r)
                    # return self.get_intention(r), response
                    return None, self.trick(query)
                if  not given_slot and given_slot is not None and given_slot is not '':
                    print 'given:', given_slot
                    given = self.graph.get_global_node(given_slot)
                else:
                    given = self.graph
                node = self.travel(tokens, given)
                self.state_cleared = False
                if node != given:
                    if self.last_slot == given.slot:
                        url = self.base_url + "&q=exact_intention:" + node.slot
                    else:
                        url = self.base_url + "&q=exact_last_intention:" + self.last_slot + "%20AND%20exact_intention:" + node.slot
                    print "1",url
                    r = requests.get(url)
                    if self.num_answer(r) > 0:
                        self.last_slot = node.slot
                        self.should_clear_state(node.slot)
                        print 'clear deepest_', node.slot, self.get_response(r)
                        return node.slot, self.get_response(r)
                else:
                    return None, self.trick(query)

        else:
            if  not given_slot and given_slot is not None and given_slot is not '':
                self.last_slot = given_slot
            else:
                if not self.last_slot:
                    self.clear_state()
                    print 'restart'
                    slot, response = self.r_walk_with_pointer(query)
                    return slot, response
            url = self.base_url + "&q=exact_question:" + query + "%20AND%20exact_last_intention:" + self.last_slot
            print url
            r = requests.get(url)

            if self.num_answer(r) > 0:
                self.state_cleared = False
                self.last_slot = slot_ = self.get_intention(r)
                self.should_clear_state(self.last_slot)
                print 'non clear exact_', slot_, self.get_response(r)
                return slot_, self.get_response(r)
            else:
                ## first try fuzzy query
                cleansed = self.cleanse(query)
                url = self.base_url + "&q=question:" + cleansed + "%20AND%20exact_last_intention:" + self.last_slot
                print "fuzzy2:", url
                if not cleansed:
                    self.clear_state()
                    return self.r_walk_with_pointer(query)
                r = requests.get(url)
                if self.num_answer(r) > 0:
                    self.state_cleared = False
                    self.last_slot = slot_ = self.get_intention(r)
                    self.should_clear_state(self.last_slot)

                    return slot_, self.get_response(r)

                node = self.graph.all_nodes[self.last_slot]
                next_node = None
                tks = self.tokenize(query)
                if not tks:
                    self.clear_state()
                    return self.r_walk_with_pointer(query)
                tokens = tks.split(" ")
                _tokens = set()
                for t in tokens:
                    _tokens.add(t)
                # key_found = False
                # num_found = False
                # for t in tokens:
                #     ss = t.split("/")
                #     word = ss[0]
                #     tag = ss[1]
                #     value_types = node.value_types
                #     if "KEY" in value_types and tag != "CD":
                #         next_node = node.go(q=word, value_type="KEY")
                #         key_found = next_node is not None
                #     if "RANGE" in value_types and tag == "CD":
                #         next_node = node.go(q=float(word), value_type="RANGE")
                #         num_found = next_node is not None
                next_node = self.travel(_tokens, node)
                if next_node == node:
                    ## query solr
                    self.clear_state()
                    return self.r_walk_with_pointer(query)
                else:
                    url = self.base_url + "&q=last_intention:" + self.last_slot + "%20AND%20intention:" + next_node.slot
                    print url
                    r = requests.get(url)

                if self.num_answer(r) > 0:
                    response = self.get_response(r)
                    slot_ = self.get_intention(r)
                    self.last_slot = slot_
                    print type(slot_)
                    cn_util.print_cn(slot_)
                    self.state_cleared = False
                    ## but
                    # cn_util.print_cn(str(self.graph.get_global_node(slot).classified_out_neighbors))
                    self.should_clear_state(slot_)
                    print 'non clear deepest _', slot_, self.get_response(r)
                    return slot_, self.get_response(r)
                else:
                    ## do trick
                    self.clear_state()
                    url = self.concat_solr_request(query=query, base_url=self.trick_url)
                    r = requests.get(url)
                    response = self.get_response(r)
                    print "None-CLEAR-Trick", self.get_response(r)
                    return None, response

                    ## tokens: set

    def travel_with_clf(self, tokens, node, query, gbdt_recursion = True):
        key = None  ## word/tag
        next_node = None
        key_found = False
        num_found = False
        value_types = node.value_types
        if "RANGE" in value_types:
            for t in tokens:
                ss = t.split("/")
                word = ss[0]
                tag = ss[1]
                if tag == "CD":
                    try:
                        _word = cn2arab.cn2arab(word)
                        next_node = node.go(q=float(_word), value_type="RANGE")
                        num_found = next_node is not None
                        if num_found:
                            print('found type by RANGE:', next_node.slot)
                            self.last_slot = node.slot
                            key = word + "/" + tag
                            break
                    except:
                        num_found = False

        if not key_found and "KEY" in value_types and gbdt_recursion:
            try:
                slot, proba = self.gbdt.predict(parent_slot=node.slot, input_=query)
                next_node = self.graph.get_global_node(slot=slot)
                key_found = next_node is not None and proba > 0.85
                if key_found:
                    print('found type by gbdt:', cn_util.cn(slot), proba)
                    self.last_slot = node.slot
                else:
                    print('NOT found type by gbdt:', cn_util.cn(slot), proba)
            except Exception,e:
                print(e.message)
                key_found = False

        ## last try of KEY
        if not key_found and not num_found and "KEY" in value_types:
            for t in tokens:
                try:
                    ss = t.split("/")
                    word = ss[0]
                    tag = ss[1]
                    value_types = node.value_types
                    if "KEY" in value_types and tag != "CD":
                        try:
                            next_node = node.go(q=word, value_type="KEY")
                            key_found = next_node is not None
                            if key_found:
                                print('found type by match:', next_node.slot)
                                self.last_slot = node.slot
                                key = word + "/" + tag
                                break
                        except:
                            pass
                except:
                    key_found = False

        if not key_found and not num_found:
            ## last try solr:
            return node
        else:
            try:
                tokens.remove(key)
                word = key.split("/")[0]
                query = query.replace(word, "")
            except:
                pass
            finally:
                if len(tokens) == 0 or query == '':
                    return next_node
                print key, next_node.slot, node.slot
                # return self.travel_with_clf(tokens, next_node, query)
                if num_found:
                    # return self.travel(tokens, next_node)
                    return self.travel_with_clf(tokens=tokens, node=next_node, query=query, gbdt_recursion=True)
                else:
                    # return self.travel(tokens=tokens, node=next_node)
                    return self.travel_with_clf(tokens=tokens, node=next_node, query=query, gbdt_recursion=not gbdt_recursion)

    def r_walk_with_pointer_with_clf(self, query, given_slot=None):
        r = None
        response = None
        if self.state_cleared:
            if given_slot:
                self.should_clear_state(self.last_slot)
                url = self.base_url + "&q=exact_question:" + query + "%20AND%20exact_intention:" + given_slot
            else:
                url = self.base_url + "&q=exact_question:" + query
            print url
            r = requests.get(url)

            if self.num_answer(r) > 0:
                self.state_cleared = False
                return_slot = self.last_slot = self.get_intention(r)
                self.should_clear_state(self.last_slot)
                return_response = self.get_response(r)
                print 'clear exact_', return_slot, self.get_response(r)
                return return_slot, return_response
            else:
                tokenized = self.tokenize(query)
                tokens = tokenized.split(" ")
                print tokens, tokenized
                if not tokenized:
                    ## do trick
                    self.clear_state()
                    return None, self.trick(query)
                if given_slot:
                    print 'given:', given_slot
                    given = self.graph.get_global_node(given_slot)
                else:
                    given = self.graph
                node = self.travel_with_clf(tokens, given, query)
                self.state_cleared = False
                if node != given:
                    if self.last_slot == given.slot:
                        url = self.base_url + "&q=exact_intention:" + node.slot
                    else:
                        url = self.base_url + "&q=exact_last_intention:" + self.last_slot + "%20AND%20exact_intention:" + node.slot
                    print "1",url
                    r = requests.get(url)
                    if self.num_answer(r) > 0:
                        self.last_slot = node.slot
                        self.should_clear_state(node.slot)
                        print 'clear deepest_', node.slot, self.get_response(r)
                        return node.slot, self.get_response(r)
                else:
                    return None, self.trick(query)

        else:
            if given_slot:
                self.last_slot = given_slot
            else:
                if not self.last_slot:
                    self.clear_state()
                    print 'restart'
                    slot, response = self.r_walk_with_pointer_with_clf(query)
                    return slot, response
            url = self.base_url + "&q=exact_question:" + query + "%20AND%20exact_last_intention:" + self.last_slot
            print url
            r = requests.get(url)

            if self.num_answer(r) > 0:
                self.state_cleared = False
                self.last_slot = slot_ = self.get_intention(r)
                self.should_clear_state(self.last_slot)
                print 'non clear exact_', slot_, self.get_response(r)
                return slot_, self.get_response(r)
            else:

                node = self.graph.all_nodes[self.last_slot]
                next_node = None
                tks = self.tokenize(query)
                if not tks:
                    self.clear_state()
                    return self.r_walk_with_pointer_with_clf(query)
                tokens = tks.split(" ")
                _tokens = set()
                for t in tokens:
                    _tokens.add(t)
                next_node = self.travel_with_clf(_tokens, node, query)
                if next_node == node:
                    ## query solr
                    self.clear_state()
                    return self.r_walk_with_pointer_with_clf(query)
                else:
                    url = self.base_url + "&q=last_intention:" + self.last_slot + "%20AND%20intention:" + next_node.slot
                    print url
                    r = requests.get(url)

                if self.num_answer(r) > 0:
                    response = self.get_response(r)
                    slot_ = self.get_intention(r)
                    self.last_slot = slot_
                    self.state_cleared = False
                    ## but
                    cn_util.print_cn(str(self.graph.get_global_node(slot_).classified_out_neighbors))
                    self.should_clear_state(slot_)
                    print 'non clear deepest _', slot_, self.get_response(r)
                    return slot_, self.get_response(r)
                else:
                    ## do trick
                    self.clear_state()
                    url = self.concat_solr_request(query=query, base_url=self.trick_url)
                    r = requests.get(url)
                    response = self.get_response(r)
                    print "None-CLEAR-Trick", self.get_response(r)
                    return None, response

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
        return None

    def should_clear_state(self, slot):
        try:
            node = self.graph.get_global_node(slot)
            if len(node.classified_out_neighbors) == 0:
                self.clear_state()
        except:
            self.clear_state()

    def get_response(self, r, i = 0):
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
            url = url + "&fq=amount_upper:[" + str(num_slot) + " TO *]" + "&fq=amount_lower:[0 TO " + str(num_slot) + "]"



        print url

        return url

    def cut(self, input_):
        # return self.tokenize(input_)
        return self.jieba_cut(input_)
        # return self.stanford_web_cut(input_)

    remove_tags = ["PN", "VA", "AD"]
    tokenizer_url = "http://localhost:11415/pos?q="
    def stanford_web_cut(self, q):
        r = requests.get(url=self.tokenizer_url + q)
        ## purify
        text = []
        for t in r.text.encode("utf-8").split(" "):
            tag = t.split("/")[1]
            word = t.split("/")[0]
            if not tag in self.remove_tags:
                text.append(word)
        return "#".join(text)

    def jieba_cut(self, input_):
        seg = "#".join(jieba.cut(input_, cut_all=False))
        tokens = _uniout.unescape(str(seg), 'utf8')
        return tokens

    remove_tags = ["PN","VA","AD"]
    def tokenize(self, q):
        r = requests.get(url=self.tokenizer_url+q)
        ## purify
        text = ""
        for t in r.text.encode("utf-8").split(" "):
            tag = t.split("/")[1]
            if not tag in self.remove_tags:
                text = text + " " + t
        text = text.strip()
        return text

    def cleanse(self, q):
        r = requests.get(url=self.tokenizer_url + q)
        ## purify
        text = ""
        for t in r.text.encode("utf-8").split(" "):
            tag = t.split("/")[1]
            if not tag in self.remove_tags:
                text = text + " " + t
        text = text.strip()
        return text

    def pure_tokenize(self, q):
        tokens = self.tokenize(q).split(" ")
        new_tokens = tokens[0].split("/")[0]
        for i in xrange(1, len(tokens)):
            new_tokens = new_tokens + " " + tokens[i].split("/")[0]
        return new_tokens

if __name__ == "__main__":
    K = GKernel("../model/graph.pkl", "../model/seq_clf.pkl")
    while 1:
        ipt = raw_input()
        print str(K.tokenize(ipt))
        tt = ipt.split(",")
        ##response = K.kernel(ipt)
        s = None
        if len(tt) >= 2:
            q = tt[0]
            s = tt[1]
        q = tt[0]
        print q, s
        if s:
            next_slot, response = K.r_walk_with_pointer_with_clf(q, s.encode('utf8'))
        else:
            next_slot, response = K.r_walk_with_pointer_with_clf(q)
        print(str(response))