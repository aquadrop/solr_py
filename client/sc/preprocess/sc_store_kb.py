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

data_path = '../../../data/sc/store_kb.txt'
output = '../../../data/sc/solr/store_kb.json'


def get_pruned(data_path, output):
    with open(output, 'w+') as out:
        with open(data_path, 'r') as inp:
            for i, l in enumerate(inp):
                try:
                    name, labels,location = l.strip().split('#')
                    d = dict()
                    d['name'] = name.split('|')
                    d['label'] = labels.split(',')
                    d['location'] = location.split('|')
                    out.write(json.dumps(d, ensure_ascii=False) + '\n')
                except Exception,e:
                    print(e.message)


if __name__ == '__main__':
    get_pruned(data_path, output)
