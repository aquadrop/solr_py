#!/usr/bin/env python
# -*- coding: utf-8 -*-
from node import Node
import cPickle as pickle
import _uniout

class Graph(Node, object):

    def __init__(self):
        super(Graph, self).__init__(slot="#ROOT#")
        ## all_nodes keyed as slots' names
        self.all_nodes = {}
        self.value_type = "KEY"
        self.value_types.add(self.value_type)

    def add_node(self, node):
        if not self.classified_out_neighbors.has_key(self.value_type):
            self.classified_out_neighbors[self.value_type] = {}
        self.classified_out_neighbors[self.value_type][node.slot] = node
        for slot in node.slot_syno:
            self.classified_out_neighbors[self.value_type][slot] = node

    def get_neighbor_node(self, slot):
        # print _uniout.unescape(str(self.classified_out_neighbors[self.value_type][slot]), 'utf8')
        return self.classified_out_neighbors[self.value_type][slot]

    def get_global_node(self, slot):
        # print _uniout.unescape(str(self.classified_out_neighbors[self.value_type][slot]), 'utf8')
        return self.all_nodes[slot]

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            graph = pickle.load(f)
            return graph

if __name__ == "__main__":
    graph = Graph.load("../model/graph.pkl")
    graph.get_neighbor_node("取款")