#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json

reload(sys)
sys.setdefaultencoding("utf-8")

data_path = '../../data/sc/dialogue.txt'
output = '../../data/sc/pruned_dialogue.txt'


def get_pruned(data_path, output):
    with open(output, 'w+') as out:
        with open(data_path, 'r') as inp:
            for l in inp:
                line = json.loads(l.strip().decode('utf-8'))
                super_intention = line['super_intention']
                super_intention = 'ROOT' if super_intention == '' else super_intention
                intention = line['intention']
                question = line['question']

                res = super_intention + ',' + intention + '\t' + question
                out.write(res + '\n')


if __name__ == '__main__':
    get_pruned(data_path, output)
