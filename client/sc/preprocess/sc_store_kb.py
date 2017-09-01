#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
import os

reload(sys)
sys.setdefaultencoding("utf-8")

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

from client.cn_util import print_cn

data_path = '../../../data/sc/raw/item.txt'
output = '../../../data/sc/solr/item.json'


def get_pruned(data_path, output):
    with open(output, 'w+') as out:
        with open(data_path, 'r') as inp:
            for i, l in enumerate(inp):
                name = l.strip()
                d = dict()
                d['name'] = name.split('|')
                d['type'] = "item"
                d['label'] = []
                d['location'] = []
                d['definition'] = []
                d['time'] = []
                d['application'] = []
                d['listing'] = []
                out.write(json.dumps(d, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    get_pruned(data_path, output)
