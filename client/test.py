#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
import _uniout
import requests

url = "http://localhost:11303/bot?q="
clear = "http://localhost:11303/clear"

correct = 0.0
total = 0.0

input_path = "../data/data.txt"

def train_error():
    global correct
    global total
    line_num = 0
    with open(input_path, 'r') as f:
        for line in f:
            text = json.loads(line)
            question_list = text["question_list"]
            answer_list = text["answer_list"]
            for i in xrange(0, len(question_list)):
                p_url = url + question_list[i]
                r = requests.get(p_url)
                total = total + 1.0
                if answer_list[i] == r.text:
                    correct = correct + 1.0
                else:
                    print question_list[i], answer_list[i], r.text
                    print line

            requests.get(clear)
            line_num = line_num + 1
            print "##", line_num

        print "correct", correct/total

if __name__ == '__main__':
    train_error()