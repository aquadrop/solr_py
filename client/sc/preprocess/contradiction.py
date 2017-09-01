#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
import os

reload(sys)
sys.setdefaultencoding("utf-8")
import traceback
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

from client.cn_util import print_cn

data_path = '../../../data/sc/train/sale_train0831.txt'


def get_pruned(data_path):
        with open(data_path, 'r') as inp:
            query_to_intention = dict()
            for i, l in enumerate(inp):
                try:
                    intention, query = l.decode('utf-8').split('#')
                    if query not in query_to_intention:
                        query_to_intention[query] = intention
                    else:
                        _intention = query_to_intention[query]
                        if intention != _intention:
                            print_cn(query)

                except Exception,e:
                    traceback.print_exc()


if __name__ == '__main__':
    get_pruned(data_path)
