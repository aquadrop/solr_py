#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" A neural chatbot using sequence to sequence model with
attentional decoder.

This is based on Google Translate Tensorflow model
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/

Sequence to sequence model by Cho et al.(2014)

Created by Chip Huyen as the starter code for assignment 3,
class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu

This file contains the code to run the model.

See readme.md for instruction on how to run the starter code.

This implementation learns NUMBER SORTING via seq2seq. Number range: 0,1,2,3,4,5,EOS

https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf

See README.md to learn what this code has done!

Also SEE https://stackoverflow.com/questions/38241410/tensorflow-remember-lstm-state-for-next-batch-stateful-lstm
for special treatment for this code
"""
from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import sys
import time
import inspect
import json

import uuid

import re

import unicodedata

reload(sys)
sys.setdefaultencoding("utf-8")

topic_sign = ['一.','二.','三.','四.','五.','六.','七.']
talk_sign = r'^[0-9]+.*$'
talk_pattern = re.compile(talk_sign)
guest_sign = r'G:.*'
guest_pattern = re.compile(guest_sign)
bot_sign = r'B:.*'
bot_pattern = re.compile(bot_sign)

def get_topic(line):
    tt = []
    start = False
    for c in line:
        if c == ':':
            start = True
            continue
        if start:
            tt.append(c)
    return ''.join(tt)

def topic_start(line):
    return '话题:' in line

def interactive(file_, write_file_):
    D = []
    with open(file_, 'rb') as f:
        data = dict()
        for line in f:
            line = line.strip().decode('utf-8')
            line = unicodedata.normalize('NFKC', line)
            if topic_start(line):
                data = dict()
                line = get_topic(line).strip().replace(" ", "").replace("\t", "")
                topic = line
                continue
            if talk_pattern.match(str(line)):
                line = re.sub('^[0-9]+(.)', '', str(line)).strip()
                if len(data) > 0:
                    if 'b' in data and len(data['b']) > 0:
                        D.append(data)
                data = dict()
                data['b'] = []
                data['topic'] = topic
                data['id'] = str(uuid.uuid4())
                if guest_pattern.match(str(line)):
                    data['g'] = str(line).replace('G:', '')
                elif bot_pattern.match(str(line)):
                    data['b'].append(str(line).replace('B:', ''))
                else:
                    data['g'] = str(line).replace('G:', '')
                continue
            if guest_pattern.match(str(line)):
                if 'g' in data:
                    last_g = data['g']
                if len(data) > 0:
                    if 'b' in data and len(data['b']) > 0:
                        D.append(data)
                data = dict()
                if last_g:
                    data['last_g'] = last_g
                data['b'] = []
                data['topic'] = topic
                data['id'] = str(uuid.uuid4())
                data['g'] = str(line).replace('G:','')
            if bot_pattern.match(str(line)):
                data['b'].append(str(line).replace('B:',''))
        if len(data) > 0:
            if 'b' in data and len(data['b']) > 0:
                D.append(data)
    with open(write_file_,'w') as f:
        json.dump(D, f, ensure_ascii=False)


if __name__ == '__main__':
	# interactive('../data/interactive/整理后的客服接待语料.txt','../data/interactive/interactive-all.json')
	interactive('../data/interactive/2017互动话术汇总版4.10.txt','../data/interactive/interactive2017.json')