#!/usr/bin/env python
# -*- coding: utf-8 -*-
from node import Node
import cPickle as pickle
import _uniout


class BeliefGraph(Node, object):

    def __init__(self):
        super(BeliefGraph, self).__init__(slot="ROOT")
        # all_nodes keyed as slots' names
        self.all_nodes = {}
        self.value_types = set()

    # def add_node(self, node):
    #     if not self.classified_out_neighbors.has_key(self.value_type):
    #         self.classified_out_neighbors[self.value_type] = {}
    #     self.classified_out_neighbors[self.value_type][node.slot] = node
    #     for slot in node.slot_syno:
    #         self.classified_out_neighbors[self.value_type][slot] = node
    #
    # def get_neighbor_node(self, slot):
    #     # print
    #     # _uniout.unescape(str(self.classified_out_neighbors[self.value_type][slot]),
    #     # 'utf8')
    #     return self.classified_out_neighbors[self.value_type][slot]

    # \xe5\x8f\x96\xe6\xac\xbe\xe4\xba\x8c\xe4\xb8\x87\xe4\xbb\xa5\xe4\xb8\x8b'
    def get_global_node(self, slot):
        # print
        # _uniout.unescape(str(self.classified_out_neighbors[self.value_type][slot]),
        # 'utf8')
        node = self.all_nodes[slot]
        return node

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            graph = pickle.load(f)
            return graph


def build_belief_graph(path, output):
    belief_graph = BeliefGraph()
    all_nodes = dict()
    all_nodes['ROOT'] = belief_graph
    ## first round
    with open(path, "rb") as f:
        for line in f.readlines():
            try:
            # print(line)
                parent, typed_children = line.strip('\n').split("#")
                category, children = typed_children.split(":")
                children = children.split(",")
            except:
                pass

            if not all_nodes.has_key(parent):
                node1 = Node(slot=parent)
                if parent != 'ROOT':
                    all_nodes[parent] = node1
            for child in children:
                if not all_nodes.has_key(child):
                    node2 = Node(slot=child)
                    all_nodes[child] = node2

            for child in children:
                if parent != 'ROOT':
                    _node1 = all_nodes[parent]
                    _node2 = all_nodes[child]
                    _node1.add_node(_node2, "KEY", [child])
                else:
                    _node2 = all_nodes[child]
                    belief_graph.add_node(_node2, "KEY", [child])

    belief_graph.all_nodes = all_nodes

    # build identity for each non-root node
    def build_identity(cat_dict, belief_graph):
        pass

    with open(output, 'wb') as pickle_file:
        pickle.dump(belief_graph, pickle_file, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    build_belief_graph('../data/sc/belief_graph.txt', '../model/sc/belief_graph.pkl')
