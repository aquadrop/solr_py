#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import Queue

parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parentdir)
from cn_util import print_cn

reload(sys)
sys.setdefaultencoding('utf-8')


class Node:
    def __init__(self, root, children=None):
        self.root = root
        self.children = children


class Tree:
    def __init__(self):
        self.root = Node('ROOT')
        self.pre_order_list = list()

    def add(self, node):
        if node == None:
            return
        queue = Queue.Queue()
        queue.put(self.root)
        while (not queue.empty()):
            n = queue.get()
            if n.root == node.root:
                if n.children == None:
                    n.children = list()
                n.children.extend(node.children)
                break
            if n.children != None:
                for item in n.children:
                    queue.put(item)

    def pre_order(self, node):
        if node == None:
            return
        self.pre_order_list.append(node.root)
        # print_cn(node.root)

        if node.children:
            for n in node.children:
                self.pre_order(n)


def get_node(line):
    parent = line.split('#')[0]
    children = line.split('#')[1].split(':')[2].strip()
    children = children.split(',')
    for i, item in enumerate(children):
        children[i] = Node(children[i])
    return Node(parent, children)


def build_tree(data_path):
    tree = Tree()
    with open(data_path, 'r') as inp:
        for line in inp:
            node = get_node(line)
            tree.add(node)

    return tree

def sort_intention(input_):
    tree = build_tree('../../../data/sc/belief_graph.txt')
    tree.pre_order(tree.root)
    pre_order_list = tree.pre_order_list
    input_=input_.split(',')
    sorted_intention = sorted(input_, key=lambda x:pre_order_list.index(x))
    return sorted_intention


if __name__ == '__main__':
    test=['女,购物,衣服','吃饭,低,有','辣,吃饭,低,有,女,购物,衣服']
    for t in test:
        print_cn(sort_intention(t))

