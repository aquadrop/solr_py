#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
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
    data_path='../../data/sc/train/sale_train0824.txt'
    encoder_vocab_path='../../data/sc/train/belief_rnn/encoder_vocab.txt'
    decoder_vocab_path='../../data/sc/train/belief_rnn/encoder_vocab.txt'
    fasttext_path='../../data/sc/train/belief_rnn/fasttext_wv.txt'
    # word2vec_path='../../data/sc/belief_rnn/encoder_vocab.txt'
    files = [encoder_vocab_path, decoder_vocab_path, fasttext_path]
    maybe_process_data(data_path,files)


# def padding2(inputs):
    #     batch_size = len(inputs)
    #     sequence_lengths = [len(seq) for seq in inputs]
    #     max_sequence_length = max(sequence_lengths)
    #
    #     result = list()
    #     for input in inputs:
    #         input.extend(['#PAD#'] * (max_sequence_length - len(input)))
    #         result.append(input)
    #
    #     return result

    #
    # def fasttext_vector(tokens):
    #     global url
    #     ff_url = url + ','.join(tokens).decode('utf-8')
    #     r = requests.get(url=ff_url)
    #     vector = r.json()['vector']
    #     return vector

    #
    # def embedding(inputs, train=True):
    #     embeddings = list()
    #     for inp in inputs:
    #         embedding = list()
    #         for word in inp:
    #             if train:
    #                 resp = wv.get(word.strip().encode('utf-8'), fasttext_wv(word))
    #                 # if not resp:
    #                 #     continue
    #             else:
    #                 resp = fasttext_wv(word)
    #             embedding.append(resp)
    #         embeddings.append(embedding)
    #     return np.squeeze(np.asarray(embeddings))


    # def add_extra_dict(path):
    #     extra_words = []
    #     with open(path, 'r') as inp:
    #         for line in inp:
    #             word = line.strip().replace(' ', '').encode('utf-8')
    #             if word not in extra_words:
    #                 extra_words.append(word)
    #                 jieba.add_word(word)
    #     return extra_words

    # def wv2memory(path):
    #     out = open('../data/sc/ooooooooout.txt', 'w')
    #     with open(path, 'r') as f:
    #         for line in f:
    #             ln = line.split('#')[1].strip().encode('utf-8')
    #             label = line.split('#')[0].split(',')
    #
    #             tokens = cut(ln)
    #             for word in tokens:
    #                 word = word.encode('utf-8')
    #                 if word not in wv:
    #                     # print_out(word, out)
    #                     embedding = fasttext_wv(word)
    #                     wv[word] = embedding
    #             for word in label:
    #                 word = word.encode('utf-8')
    #                 if word not in wv:
    #                     # print_out(word, out)
    #                     embedding = fasttext_wv(word)
    #                     wv[word] = embedding
    #     wv["#PAD#"] = fasttext_wv("#PAD#")
    #     wv["#EOS#"] = fasttext_wv("#EOS#")
    #     wv["#GO#"] = fasttext_wv("#GO#")
    #     wv["#UNK#"] = fasttext_wv("#UNK#")
    #     wv['/'] = fasttext_wv('/')
    #     # wv['\n']=fasttext_wv('\n')
    #     # print(len(wv))

    # def load_fasttext2(data_path, extra_word_path):
    #     extra_words = add_extra_dict(extra_word_path)
    #     vocab = []
    #     with open(data_path, 'r') as f:
    #         for line in f:
    #             line = line.strip().replace(' ', '').encode('utf-8')
    #             query = line.split('#')[1]
    #             label = line.split('#')[0].split(',')
    #             tokens = cut(query)
    #             tokens.extend(label)
    #             for word in tokens:
    #                 word = word.encode('utf-8')
    #                 if word not in vocab:
    #                     vocab.append(word)
    #     vocab.extend(extra_words)
    #     vocab = list(set(vocab))
    #
    #     embeddings = []
    #     for w in vocab:
    #         embedding = fasttext_wv(w)
    #         embeddings.append(embedding)
    #
    #     return vocab, embeddings

    #
    # def init_dict(dict_path):
    #     label2index_f = open(dict_path[0], 'r')
    #     index2label_f = open(dict_path[1], 'r')
    #
    #     global label2index
    #     label2index = json.load(label2index_f)
    #     global index2label
    #     index2label = json.load(index2label_f)

    # def recover(index, vocab, source=True):
    #     sentence = []
    #     ignore = ['#PAD#', '#UNK#', '#GO#', 0, 2, 3]
    #     for ii in index:
    #         if ii in ignore:
    #             continue
    #         if ii in ['#EOS#', 1]:
    #             break
    #         if source:
    #             sentence.append(vocab[ii])
    #         else:
    #             sentence.append(str(index2label[str(ii)]))
    #     return ''.join(sentence)