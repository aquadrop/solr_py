#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json

reload(sys)
sys.setdefaultencoding("utf-8")

inpath = '../../../data/sc/big.txt'
char2index_path = '../../../data/sc/char2index_dict_big.txt'
index2char_path = '../../../data/sc/index2char_dict_big.txt'


def build_dict(inpath, outpath1, outpath2):
    char2index_dict = dict()
    index2char_dict = dict()

    char2index_dict['#GO#'] = 0
    char2index_dict['#EOS#'] = 1
    char2index_dict['#PAD#'] = 2
    char2index_dict['#UNK#'] = 3

    index2char_dict[0] = '#GO#'
    index2char_dict[1] = '#EOS#'
    index2char_dict[2] = '#PAD#'
    index2char_dict[3] = '#UNK#'

    for i in range(4, 100):
        char2index_dict['holder' + str(i)] = i
        index2char_dict[i] = 'holder' + str(i)

    with open(inpath, 'r') as f:
        index = 100
        for line in f:
            line = line.decode('utf-8').strip('\n')
            unique_len = 0
            if len(line):
                for i, cc in enumerate(line):
                    if cc not in char2index_dict:
                        char2index_dict[cc] = index + unique_len
                        index2char_dict[index + unique_len] = cc
                        # print(index + unique_len)
                        unique_len += 1
            index += unique_len

    char2index_f = open(char2index_path, 'w')
    index2char_f = open(index2char_path, 'w')

    json.dump(char2index_dict, char2index_f, ensure_ascii=False)
    json.dump(index2char_dict, index2char_f, ensure_ascii=False)

    char2index_f.close()
    index2char_f.close()


def build_label_dict(inpath, outpath1, outpath2):
    label2index = dict()
    index2label = dict()

    label2index['#GO#'] = 0
    label2index['#EOS#'] = 1
    label2index['#PAD#'] = 2
    label2index['#UNK#'] = 3
    label2index['/'] = 4

    index2label[0] = '#GO#'
    index2label[1] = '#EOS#'
    index2label[2] = '#PAD#'
    index2label[3] = '#UNK#'
    index2label[4] = '/'

    for i in range(5, 100):
        label2index['holder' + str(i)] = i
        index2label[i] = 'holder' + str(i)

    with open(inpath, 'r') as f:
        index = 100
        for line in f:
            labels = line.split('#')[0].split(',')
            unique_len = 0
            for i, label in enumerate(labels):
                if label not in label2index:
                    label2index[label] = index + unique_len
                    index2label[index + unique_len] = label
                    unique_len += 1
            index += unique_len
    label2index_f = open(outpath1, 'w')
    index2label_f = open(outpath2, 'w')

    json.dump(label2index, label2index_f, ensure_ascii=False)
    json.dump(index2label, index2label_f, ensure_ascii=False)

    label2index_f.close()
    index2label_f.close()


def int2str_key(dic):
    new_dict = dict()
    for key in dic.keys():
        new_dict[int(key)] = dic[key]

    return new_dict


def add(inpath, char2index_path, index2char_path):
    char2index_f = open(char2index_path, 'r')
    index2char_f = open(index2char_path, 'r')

    char2index_dict = json.load(char2index_f)
    index2char_dict = json.load(index2char_f)

    index2char_dict = int2str_key(index2char_dict)

    char2index_f.close()
    index2char_f.close()

    with open(inpath, 'r') as f:
        index = len(index2char_dict)
        for line in f:
            line = line.decode('utf-8').strip('\n')
            unique_len = 0
            if len(line):
                for i, cc in enumerate(line):
                    if cc not in char2index_dict:
                        char2index_dict[cc] = index + unique_len
                        index2char_dict[index + unique_len] = cc
                        # print(index + unique_len)
                        unique_len += 1
            index += unique_len

    char2index_f = open(char2index_path, 'w')
    index2char_f = open(index2char_path, 'w')

    json.dump(char2index_dict, char2index_f, ensure_ascii=False)
    json.dump(index2char_dict, index2char_f, ensure_ascii=False)

    char2index_f.close()
    index2char_f.close()


if __name__ == '__main__':
    build_label_dict('../../../data/sc/train/sale_train0824.txt', '../../../data/sc/dict/label2index.txt',
                     '../../../data/sc/dict/index2label.txt')
    # add(inpath, char2index_path, index2char_path)
