#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests
import json
import re


class Kernel:

    last_task = None
    current_str_slot = None
    current_num_slot = 0

    base_url = "http://localhost:11403/solr/qa/select?defType=edismax&indent=on&wt=json&rows=1"
    trick_url = "http://localhost:11403/solr/trick/select?defType=edismax&indent=on&wt=json&rows=1"

    def kernel(self, query):

        self.current_str_slot, self.current_num_slot = self.parse_query_to_slot(
            query)
        if self.current_num_slot > 0:
            query = None
        url = self.concat_solr_request(query=query,
                                       base_url=self.base_url,
                                       last_intention=self.last_task,
                                       str_slot=self.current_str_slot,
                                       num_slot=self.current_num_slot)

        self.current_num_slot = 0
        r = requests.get(url)
        if self.num_answer(r) == 0:
            self.clear_state()
            url = self.concat_solr_request(
                query=query, base_url=self.trick_url)
            r = requests.get(url)
        response = self.get_response(r)
        self.last_task = self.get_intention(r)

        self.print_params()

        if not self.get_next_intention(r):
            self.clear_state()
        return response

    def clear_state(self):
        print 'state cleared'
        self.last_task = None
        self.current_num_slot = 0
        self.current_str_slot = None

    def print_params(self):
        print "last_task:", self.last_task, "current_task：" "current_str_slot：", self.current_str_slot, "current_num_slot", self.current_num_slot

    def num_answer(self, r):
        return int(r.json()["response"]["numFound"])

    def parse_query_to_slot(self, query):
        nums = re.findall('\d+', query)
        str_slot = re.sub('\d', '', query)
        num_slot = nums[0] if len(nums) > 0 else 0
        print num_slot
        return str_slot, num_slot

    def get_response(self, r):
        try:
            return r.json()["response"]["docs"][0]["answer"][0].encode('utf8')
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
    K = Kernel()
    while 1:
        ipt = raw_input()
        response = K.kernel(ipt)
        print(str(response))
