#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
import _uniout
import cPickle as pickle
from graph import Graph
from key_search_node import KeySearchNode
from range_search_node import RangeSearchNode
from node import Node

reload(sys)
sys.setdefaultencoding("utf-8")

def print_cn(q):
    print _uniout.unescape(str(q), 'utf8')

def cn(q):
    return _uniout.unescape(str(q), 'utf8')