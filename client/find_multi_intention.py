#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import _uniout
import sys
import csv

reload(sys)
sys.setdefaultencoding("utf-8")

data_path = '../data/train_pruned_fixed.txt'
classes = list()
inputs = dict()
results = dict()

if __name__ == '__main__':
    with open(data_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            a = line[0]
            if a not in classes:
                classes.append(a)
    print("classes:", _uniout.unescape(str(classes), 'utf8'))

    with open(data_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            a = line[0]
            b = line[1]
            a_slots = a.split(',')
            if b not in inputs:
                inputs[b] = []
                inputs[b].append(a)

            for x in inputs[b]:
                x_slots = x.split(',')
                if x_slots[0] == a_slots[0] and x_slots[1] != a_slots[1]:
                    inputs[b].append(a)

    for inp, intentions in inputs.iteritems():
        if len(intentions) >= 2:
            # results[inp] = intentions
            print(inp, _uniout.unescape(str(intentions), 'utf8'))


# print(_uniout.unescape(str(results), 'utf8'))
# print(_uniout.unescape(str(results['用卡取两百块']), 'utf8'))