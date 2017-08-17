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

data_path = '../../../data/sc/raw/111sale'
output = '../../../data/sc/raw/111sale_train.txt'


def get_pruned(data_path, output):
    with open(output, 'w+') as out:
        with open(data_path, 'r') as inp:
            for i, l in enumerate(inp):
                try:
                    line = json.loads(l.strip().decode("utf-8"))
                    # super_intention = line['super_intention']
                    # super_intention = ['ROOT'] if super_intention == [] else super_intention
                    intention = line['intention']
                    intention = filter(lambda x: x != 'ANY', intention)
                    # print_cn(intention)
                    # if len(intention) == 1:
                    #     intention = ['ROOT'] + intention
                    #     print_cn(intention)
                    questions = line['question']
                    intention = ','.join(intention)
                    for question in questions:
                        out.write(intention + '#' + question + '\n')
                except Exception,e:
                    print(e.message)


if __name__ == '__main__':
    get_pruned(data_path, output)
