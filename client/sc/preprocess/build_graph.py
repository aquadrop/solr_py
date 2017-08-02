#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import re

import json
import _uniout
import cPickle as pickle
from client.graph import Graph
from client.node import Node

reload(sys)
sys.setdefaultencoding("utf-8")

def preprocess(path):
    edges = set()
    with open(path, "r") as file:
        lines = file.readlines()
    texts = list()
    intention_list = list()
    for line in lines:
        text = json.loads(line)["intention_list"]
        texts.append(text)
    # print _uniout.unescape(str(texts), 'utf8')
    for intentions in texts:
        for i in xrange(len(intentions) - 1):
            edges.add(intentions[i] + "," + intentions[i + 1])

    for key in edges:
        print key

breakpoints_map = {"转账":[-10,0,49999,1000000-1,9999999999999],
                   "存款":[-10,0,99,50000-1,9999999999999],
                   "取款":[-10,0,99,20000-1,50000-1,9999999999999]}
def build_graph(path, output):
    graph = Graph()
    all_nodes = {}
    ## first round
    with open(path, "rb") as f:
        for line in f.readlines():
            # print(line)
            edge, value, _type = line.strip('\n').split(" ")
            slot1, slot2 = edge.split(",")

            if _type == "REGEX":
                if not all_nodes.has_key(slot1):
                    node1 = Node(slot=slot1)
                    all_nodes[slot1] = node1
                if not all_nodes.has_key(slot2):
                    node2 = Node(slot=slot2)
                    all_nodes[slot2] = node2
            if slot1 != 'ROOT':
                _node1 = all_nodes[slot1]
                _node2 = all_nodes[slot2]
                value = re.compile(value)
                _node1.add_node(_node2, _type, value)
            else:
                _node2 = all_nodes[slot2]
                graph.add_node(_node2)
    # with open(path, "rb") as f:
    #     for line in f.readlines():
    #         edge, value, _type = line.strip('\n').split(" ")
    #         slot1, slot2 = edge.split(",")
    #         node1 = all_nodes[slot1]
    #         if all_nodes.has_key(slot2):
    #             node2 = all_nodes[slot2]
    #         else:
    #             node2 = Node(slot=slot2, value=value, value_type=None)
    #             all_nodes[slot2] = node2
    #         node1.add_node(node2)

    graph.all_nodes = all_nodes
    # for key, node in graph.all_nodes.iteritems():
    #     ## initital nodes
    #     if len(node.classified_in_neighbors) == 0:
    #         graph.add_node(node)
    with open(output, 'wb') as pickle_file:
        pickle.dump(graph, pickle_file, pickle.HIGHEST_PROTOCOL)


def compute_intention_graph(path):
    with open(path, "r") as file:
        lines = file.readlines()
    texts = list()
    intention_list = list()
    for line in lines:
        text = json.loads(line)["intention_list"]
        texts.append(text)
    # print _uniout.unescape(str(texts), 'utf8')
    for intentions in texts:
        for intention in intentions:
            if intention not in intention_list:
                intention_list.append(intention)
    # print len(intention_list)
    # print _uniout.unescape(str(intention_list), 'utf8')
    intention_graph = dict()
    for intention in intention_list:
        key = intention
        value = list()
        for x in texts:
            if key in x and x.index(key) + 1 < len(x) and x[x.index(key) + 1] not in value:
                value.append(x[x.index(key) + 1])
        intention_graph[key] = value
    print _uniout.unescape(str(intention_graph), 'utf8')
    return intention_graph


if __name__ == '__main__':
    build_graph("/home/deep/solr/solr-6.5.1/solr_py/data/sc/graph.txt",
                "/home/deep/solr/solr-6.5.1/solr_py/model/sc_graph_v7.pkl")