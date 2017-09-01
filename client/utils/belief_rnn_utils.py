#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import random
import numpy as np
import jieba
import requests

from query_util import QueryUtils
from cn_util import print_cn
from cn_util import print_out
from itertools import chain

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

reload(sys)
sys.setdefaultencoding("utf-8")


def add_dict():
    path='../../data/sc/train/belief_rnn/decoder_vocab.txt'
    with open(path,'r') as f:
        for line in f:
            line=line.strip()
            jieba.add_word(line)
    # print_cn(cut('我要吃张氏宽窄巷和很高兴遇见你'))

def get_train_data(data_path,normal_path,shuffle_path):
    add_dict()
    normal_f=open(normal_path,'w')
    shuffle_f=open(shuffle_path,'w')
    # drop_f=open(drop_path)



    with open(data_path,'r') as f:
        for line in f:
            print_cn(line)
            line=json.loads(line)
            intentions=','.join(line['intention'])
            questions=line['question']

            for question in questions:
                normal_f.write(intentions + '#' + question + '\n')
                tokens = cut(question)
                if len(tokens)<=1:
                    continue
                elif len(tokens)==2:
                    random.shuffle(tokens)
                    shuffle_f.write(intentions + '#' + ''.join(tokens) + '\n')
                else:
                    for _ in range(2):
                        random.shuffle(tokens)
                        shuffle_f.write(intentions+'#'+''.join(tokens)+'\n')
    normal_f.close()
    shuffle_f.close()




def cut(input_):
    input_ = QueryUtils.static_remove_cn_punct(input_)
    tokens = list(jieba.cut(input_, cut_all=False))
    return tokens

def fasttext_wv(word):
    url = 'http://localhost:11425/fasttext/w2v?q='
    ff_url = url + word.decode('utf-8')
    r = requests.get(url=ff_url)
    vector = r.json()['vector']
    return vector

def maybe_process_data(data_path, files):
    # train_file=open(train_data_path,'w')
    pwd_path=os.path.abspath('.')
    paths=[os.path.join(pwd_path,f) for f in files]
    all_exist=True
    for path in paths:
        if not os.path.exists(path):
            all_exist=False
            break

    if all_exist:
        print('Data Exist.')
    else:
        print('Process Data.')
        encoder_vocab_file = open(files[0], 'w')
        decoder_vocab_file = open(files[1], 'w')
        fasttext_wv_file=open(files[2],'w')
        fasttext_embeddings={}

        querys = []
        encoder_vocab = []
        decoder_vocab = []

        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip().replace(' ', '')
                query = line.split('#')[1]
                querys.append(query)
                label = line.split('#')[0].split(',')
                for word in label:
                    if word not in decoder_vocab:
                        decoder_vocab.append(word)
        for word in decoder_vocab:
            jieba.add_word(word)

        for query in querys:
            tokens = list(cut(query))
            for word in tokens:
                if word not in encoder_vocab:
                    encoder_vocab.append(word)

        encoder_vocab.extend(['#EOS#', '#UNK#', '#PAD#', '#GO#'])
        decoder_vocab.extend(['#EOS#', '#UNK#', '#PAD#', '#GO#', '/'])

        for word in encoder_vocab:
            encoder_vocab_file.write(word)
            encoder_vocab_file.write('\n')
            if word not in fasttext_embeddings:
                fasttext_embeddings[word]=fasttext_wv(word)

        for word in decoder_vocab:
            decoder_vocab_file.write(word)
            decoder_vocab_file.write('\n')
            if word not in fasttext_embeddings:
                fasttext_embeddings[word]=fasttext_wv(word)

        json.dump(fasttext_embeddings,fasttext_wv_file,ensure_ascii=False)
        encoder_vocab_file.close()
        decoder_vocab_file.close()
        fasttext_wv_file.close()
        print('Process Done.')

def recover_source(index, vocab):
    sentence = []
    ignore = ['#PAD#', '#UNK#', '#GO#']
    for ii in index:
        word = vocab[ii]
        # if word in ignore:
        #     continue
        # if ii == '#EOS#':
        #     break
        sentence.append(word)
    return ''.join(sentence)


def recover_label(index, vocab):
    sentence = []
    ignore = ['#PAD#', '#UNK#', '#GO#']
    for ii in index:
        word = vocab[ii]
        # if word in ignore:
        #     continue
        # if word == '#EOS#':
        #     break
        sentence.append(word)
    return ''.join(sentence)

def load_fasttext(encoder_vocab_path):
    print('Load encoder vocab')
    encoder_vocab = []
    embeddings = []

    with open(encoder_vocab_path) as f:
        for word in f:
            word = word.strip().replace(' ', '')
            if word not in encoder_vocab:
                encoder_vocab.append(word)
                embedding = fasttext_wv(word)
                embeddings.append(embedding)
    return encoder_vocab, embeddings


def load_decoder_vocab(decoder_vocab_path):
    print('Load decoder vocab')
    decoder_vocab = []

    with open(decoder_vocab_path) as f:
        for word in f:
            word = word.strip().replace(' ', '')
            if word not in decoder_vocab:
                decoder_vocab.append(word)
    return decoder_vocab

def padding(inputs, pad):
    batch_size = len(inputs)
    sequence_lengths = [len(seq) for seq in inputs]
    max_sequence_length = max(sequence_lengths)

    inputs_batch_major = np.ones(
        (batch_size, max_sequence_length), np.int32) * pad
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i][j] = element

    return inputs_batch_major

def generate_batch_data(data_path, encoder_vocab, decoder_vocab, batch_size=32, train=True):
    sources = list()
    targets = list()

    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip().replace(' ', '').encode('utf-8')
            target = line.split('#')[0].split(',')
            space = ['/' for _ in target]
            target = list(chain(*zip(target, space)))[:-1]
            source = line.split('#')[1]
            sources.append(source)
            targets.append(target)
    size = len(sources)

    for word in encoder_vocab:
        jieba.add_word(word)
    for word in decoder_vocab:
        jieba.add_word(word)

    while True:
        ss = np.random.randint(0, size, batch_size)

        encoder_inputs = list()
        decoder_inputs = list()
        labels = list()

        for ln in ss:
            source_tokens = list(cut(sources[ln].strip().decode('utf-8')))
            # print_cn(source_tokens)
            target_tokens = targets[ln]
            # print_cn(target_tokens)
            decoder_input = ['#GO#'] + target_tokens
            label = target_tokens + ['#EOS#']
            # print_cn(label)
            encoder_inputs.append(
                [encoder_vocab.index(word) if word in encoder_vocab else encoder_vocab.index('#UNK#') for word in
                 source_tokens])
            decoder_inputs.append(
                [decoder_vocab.index(word) if word in decoder_vocab else decoder_vocab.index('#UNK#') for word in
                 decoder_input])
            labels.append(
                [decoder_vocab.index(word) if word in decoder_vocab else decoder_vocab.index('#UNK#') for word in
                 label])

        encoder_inputs_length = [len(line) for line in encoder_inputs]
        decoder_inputs_length = [len(line) for line in decoder_inputs]

        # print(encoder_inputs)
        # print(labels)
        # print('----------------------------------------------')

        encoder_inputs = padding(encoder_inputs, encoder_vocab.index('#PAD#'))
        decoder_inputs = padding(decoder_inputs, decoder_vocab.index('#PAD#'))
        labels = padding(labels, decoder_vocab.index('#PAD#'))

        # encoder_embed_inputs = embedding(encoder_inputs, train)
        # decoder_embed_inputs = embedding(decoder_inputs, train)
        # labelss = [[label2index.get(word.decode('utf-8'), 3) for word in ln] for ln in labels]

        yield np.array(encoder_inputs), np.array(decoder_inputs), np.array(labels), np.asarray(
            encoder_inputs_length), np.asarray(decoder_inputs_length)

if __name__ == '__main__':
    # data_path='../../data/sc/train/sale_train0824.txt'
    # encoder_vocab_path='../../data/sc/train/belief_rnn/encoder_vocab.txt'
    # decoder_vocab_path='../../data/sc/train/belief_rnn/encoder_vocab.txt'
    # fasttext_path='../../data/sc/train/belief_rnn/fasttext_wv.txt'
    # # word2vec_path='../../data/sc/belief_rnn/encoder_vocab.txt'
    # files = [encoder_vocab_path, decoder_vocab_path, fasttext_path]
    # maybe_process_data(data_path,files)
    # add_dict()
    get_train_data('../../data/sc/0831/111sale','../../data/sc/train/sale_train0831.txt','../../data/sc/train/sale_train0831_shuffle.txt')