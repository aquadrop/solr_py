#!/usr/bin/env python
# -*- coding: utf-8 -*-
import bisect
import _uniout

class Node:
    def __init__(self, slot, breakpoints=None, slot_syno = []):
        self.slot = slot  ## use as to query solr
        self.slot_syno = slot_syno
        self.description = None  ## range description
        # ## define edges
        # value_type = None ## KEY, RANGE, BOOL, DATE
        #
        # ...
        self.value_types = set()

        # mainly use value to as dictionary key
        # might use breakpoint, the query time is log(N)
        # neighbors = {}
        self.breakpoints = breakpoints
        self.classified_out_neighbors = {}
        self.classified_in_neighbors = {}

    ## value_type must be set
    def add_node(self, node, value_type, values):
        if not node.classified_in_neighbors.has_key(value_type):
            node.classified_in_neighbors[value_type] = {}
        node.classified_in_neighbors[value_type][self.slot] = self

        if not self.classified_out_neighbors.has_key(value_type):
            self.classified_out_neighbors[value_type] = {}

        ## synonym consideration
        for value in values:
            self.classified_out_neighbors[value_type][value] = node

        self.value_types.add(value_type)

    def go(self, q, value_type=None):
        return self.decide(q, value_type)

    def decide(self, q, value_type=None):
        try:
            if value_type == "KEY":
                return self.classified_out_neighbors[value_type][q]
            if value_type == "RANGE":
                return Node.grade(float(q), self.breakpoints, self.classified_out_neighbors[value_type])
        except Exception,e:
            print e.message
            return None
    # def grade(score, breakpoints=[60, 70, 80, 90], grades='FDCBA'):
    #     i = bisect(breakpoints, score)
    #     return grades[i]

    @staticmethod
    def grade(score, breakpoints, nodes):
        i = bisect.bisect(breakpoints, score)
        print i, nodes,breakpoints
        return nodes[i - 2]

if __name__ == "__main__":
    bp = [-10, 0, 20000, 50000, 9999999999999]
    nodes = {-1:"W",0:"两位以下",1:"两万到五万",2:"五万以上"}
    print Node.grade(30000, bp, nodes)