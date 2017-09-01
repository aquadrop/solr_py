#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import json
import jieba
import requests
import argparse
from cn_util import print_cn
from cn_util import print_out

from query_util import QueryUtils
from itertools import chain

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

reload(sys)
sys.setdefaultencoding("utf-8")

import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np
from utils.mail import send_mail

VOCAB_SIZE = 0
EMBEDDING_SIZE = 300
HIDDEN_UNIT = 128
N_LAYER = 2
BATCH_SIZE = 32
RNN_SIZE = 128


def build_vocab(inpath, outpath):
    out = open(outpath, 'w')
    l = ['#GO#', '#EOS#', '#PAD#', '#UNK#']
    for w in l:
        out.write(w.encode('utf-8'))
        out.write('\n')
    vocabs = []
    with open(inpath, 'r') as f:
        for line in f:
            line = line.strip().decode('utf-8')
            for w in line:
                if w not in vocabs:
                    vocabs.append(w)
                    out.write(w.encode('utf-8'))
                    out.write('\n')
    for i in range(100):
        out.write(('holder' + str(i)).encode('utf-8'))
        out.write('\n')
    out.close()


def init_dict(dict_path):
    char2index_f = open(dict_path[0], 'r')
    index2char_f = open(dict_path[1], 'r')

    char2index = json.load(char2index_f)
    index2char = json.load(index2char_f)

    return char2index, index2char


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


def generate_batch_data(data_path, char2index, batch_size=32, train=True):
    sources = list()
    targets = list()

    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip().replace(' ', '').decode('utf-8')
            target = line.split('#')[0]
            source = line.split('#')[1]
            sources.append(source)
            targets.append(target)
    size = len(sources)

    while True:
        ss = np.random.randint(0, size, batch_size)

        encoder_inputs = list()
        decoder_inputs = list()
        labels = list()

        for ln in ss:
            encoder_inputs.append([char2index.get(c, 3) for c in sources[ln]])
            decoder_inputs.append([char2index.get('#GO#')] + [char2index.get(c, 3) for c in targets[ln]])
            labels.append([char2index.get(c, 3) for c in targets[ln]] + [char2index.get('#EOS#')])

        encoder_inputs_length = [len(line) for line in encoder_inputs]
        decoder_inputs_length = [len(line) for line in decoder_inputs]

        # print(encoder_inputs)
        # print(labels)
        # print('----------------------------------------------')

        encoder_inputs = padding(encoder_inputs, char2index.get('#PAD#'))
        decoder_inputs = padding(decoder_inputs, char2index.get('#PAD#'))
        labels = padding(labels, char2index.get('#PAD#'))

        # encoder_embed_inputs = embedding(encoder_inputs, train)
        # decoder_embed_inputs = embedding(decoder_inputs, train)
        # labelss = [[label2index.get(word.decode('utf-8'), 3) for word in ln] for ln in labels]

        yield np.array(encoder_inputs), np.array(decoder_inputs), np.array(labels), np.asarray(
            encoder_inputs_length), np.asarray(decoder_inputs_length)


class BeliefRnn:
    def __init__(self, vocab, trainable=True):
        self.trainable = trainable
        self.vocab = vocab

    def _create_placeholder(self):
        print('Create placeholders...')

        self.labels_ = tf.placeholder(
            tf.int32, shape=(None, None), name='labels_')

        with tf.variable_scope("encoder") as scope:
            self.encoder_inputs = tf.placeholder(
                tf.int32, shape=(None, None), name="encoder_inputs")
            self.encoder_inputs_length = tf.placeholder(
                tf.int32, shape=(None,), name="encoder_inputs_length")

        with tf.variable_scope("decoder") as scope:
            self.decoder_inputs = tf.placeholder(
                tf.int32, shape=(None, None), name="decoder_inputs")
            self.decoder_inputs_length = tf.placeholder(
                tf.int32, shape=(None,), name="decoder_inputs_length")

        self.embeddings = tf.Variable(tf.random_uniform(
            [VOCAB_SIZE, EMBEDDING_SIZE], -1.0, 1.0), dtype=tf.float32)
        self.encoder_embed_inputs = tf.nn.embedding_lookup(
            self.embeddings, self.encoder_inputs)
        self.decoder_embed_inputs = tf.nn.embedding_lookup(
            self.embeddings, self.decoder_inputs)

    def _inference(self):
        print('Create inference...')

        def get_cell(size):
            cell = tf.contrib.rnn.LSTMCell(size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            return cell

        with tf.variable_scope("encoder") as scope:
            encoder_cell = tf.contrib.rnn.MultiRNNCell([get_cell(RNN_SIZE) for _ in range(N_LAYER)])
            encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, self.encoder_embed_inputs,
                                                              sequence_length=self.encoder_inputs_length,
                                                              dtype=tf.float32)
        # with tf.variable_scope("encoder") as scope:
        #     predicting_encoder_output, predicting_encoder_state = tf.nn.dynamic_rnn(encoder_cell,
        #                                                                             self.predicting_embed_inputs,
        #                                                                             sequence_length=self.predicting_inputs_length,
        #                                                                             dtype=tf.float32)
        with tf.variable_scope("decoder") as scope:
            decoder_cell = tf.contrib.rnn.MultiRNNCell([get_cell(RNN_SIZE) for _ in range(N_LAYER)])
            output_layer = Dense(VOCAB_SIZE,
                                 kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            self.max_target_sequence_length = tf.reduce_max(
                self.decoder_inputs_length, name='max_target_len')

        with tf.variable_scope("decoder") as scope:
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=self.decoder_embed_inputs,
                                                                sequence_length=self.decoder_inputs_length,
                                                                time_major=False)

            training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                               helper=training_helper,
                                                               initial_state=encoder_state,
                                                               output_layer=output_layer)

            self.decoder_output, decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                                      impute_finished=True,
                                                                                      maximum_iterations=self.max_target_sequence_length)
            # self.training_logits_ = tf.identity(self.decoder_output.rnn_output, 'logits')

        with tf.variable_scope("decoder", reuse=True):
            # predicting_helper = tf.contrib.seq2seq.TrainingHelper(inputs=self.predicting_embed_inputs,
            #                                                       sequence_length=self.predicting_inputs_length,
            #                                                       time_major=False)
            # greedy_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embeddings, start_tokens=tf.tile(
            #     tf.constant([self.decoder_vocab.index('#GO#')], dtype=tf.int32),
            #     [BATCH_SIZE]), end_token=self.decoder_vocab.index('#EOS#'))
            greedy_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embeddings,
                                                                     start_tokens=tf.fill([BATCH_SIZE],
                                                                                          self.vocab.get(
                                                                                              '#GO#')),
                                                                     end_token=self.vocab.get('#EOS#'))
            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                                 greedy_helper,
                                                                 encoder_state,
                                                                 output_layer)
            self.predicting_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                             impute_finished=True)
            self.predicting_logits = tf.identity(self.predicting_output.sample_id, name='predictions')

    def _create_loss(self):
        print('Create loss...')

        self.training_logits_ = tf.identity(self.decoder_output.rnn_output, 'logits')
        masks = tf.sequence_mask(self.decoder_inputs_length, self.max_target_sequence_length, dtype=tf.float32,
                                 name='masks')
        self.loss = tf.contrib.seq2seq.sequence_loss(self.training_logits_, self.labels_, masks)
        self.predictions_ = tf.argmax(self.training_logits_, axis=2)

    def _create_optimizer(self):
        print('Create optimizer...')

        self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    def build_graph(self):
        print('Build graph...')

        self._create_placeholder()
        self._inference()
        if self.trainable:
            self._create_loss()
            self._create_optimizer()

    def build_graph_debug(self):
        print('Build graph...')

        self._create_placeholder()
        self._inference()
        # if self.trainable:
        #     self._create_loss()
        #     self._create_optimizer()


def _check_restore_parameters(sess, saver, model_path):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(
        os.path.dirname(model_path))
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading parameters for the model")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Initializing fresh parameters for the model")


def recover(index, index2char, source=True):
    sentence = []
    if source:
        for ii in index:
            if ii in [0, 1, 2]:
                continue
            sentence.append(index2char.get(str(ii), 3))
    else:
        for ii in index:
            if ii in [0, 2, 3]:
                continue
            if ii == 1:
                break
            sentence.append(index2char.get(str(ii), 3))
    # print(sentence)
    return ''.join(sentence)


def _get_user_input():
    """ Get user's input, which will be transformed into encoder input later """
    print("query:->", end="")
    sys.stdout.flush()
    return sys.stdin.readline()


def train(data_path, dict_path, model_path):
    print('Training...')

    char2index, index2char = init_dict(dict_path)

    global VOCAB_SIZE
    VOCAB_SIZE = len(char2index)

    gen = generate_batch_data(data_path, char2index)
    model = BeliefRnn(char2index)
    model.build_graph()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver, model_path)
        max_loss = 0.5
        i = 0
        for enci, deci, lab, encil, decil in gen:
            # print(enci[0])
            # print(deci[0])
            # print(enci_emb.shape)
            # print(deci_emb.shape)
            # print(lab[0])
            model.optimizer.run(feed_dict={model.encoder_inputs.name: enci,
                                           model.encoder_inputs_length.name: encil,
                                           model.decoder_inputs.name: deci,
                                           model.decoder_inputs_length.name: decil,
                                           model.labels_.name: lab})

            if (i + 1) % 1 == 0:
                loss, predictions, logits, c = sess.run(
                    [model.loss, model.predictions_,
                     model.training_logits_, model.labels_],
                    feed_dict={model.encoder_inputs.name: enci,
                               model.encoder_inputs_length.name: encil,
                               model.decoder_inputs.name: deci,
                               model.decoder_inputs_length.name: decil,
                               model.labels_.name: lab})

                print('---------------------------------------')
                print("step", i)
                print('question:     >', recover(enci[0], index2char))
                print('answer:       >', recover(deci[0], index2char, False))
                # print('prediction:   >', predictions[0])
                print('prediction:   >', recover(predictions[0], index2char, False))
                print('loss:         >', loss)

                if loss < max_loss:
                    max_loss = loss * 0.7
                    print('saving model...', i, loss)
                    saver.save(
                        sess, model_path, global_step=i)
                if i % 1000 == 0 and i > 100:
                    print('safe_mode saving model...', i, loss)

                    loss, predictions, logits, c = sess.run(
                        [model.loss, model.predictions_,
                         model.training_logits_, model.labels_],
                        feed_dict={model.encoder_inputs.name: enci,
                                   model.encoder_inputs_length.name: encil,
                                   model.decoder_inputs.name: deci,
                                   model.decoder_inputs_length.name: decil,
                                   model.labels_.name: lab})

                    sends = 'step ' + str(i) + '\n' \
                            + 'question:     >' + recover(enci[0], index2char) + '\n' \
                            + 'answer:       >' + recover(deci[0], index2char, False) + '\n' \
                            + 'prediction:   >' + recover(predictions[0], index2char, False) + '\n' \
                            + 'loss:         >' + str(loss)

                    send_mail(sends)
                    saver.save(
                        sess, model_path, global_step=i)
            i = i + 1


def predict(dict_path, model_path):
    char2index, index2char = init_dict(dict_path)
    global VOCAB_SIZE
    VOCAB_SIZE = len(char2index)

    model = BeliefRnn(char2index, False)
    model.build_graph()

    saver = tf.train.Saver()
    # loaded_graph = tf.Graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver, model_path)
        while True:
            line = _get_user_input()
            line = line.strip().decode('utf-8')
            ids = [char2index.get(c, 3) for c in line]

            print_cn(line)
            print(ids)
            inputs_length = [len(ids)] * BATCH_SIZE
            ids_inputs = [ids for _ in range(BATCH_SIZE)]
            # predicting_embed_inputs = [[fasttext_wv(word) for word in predicting_inputs] for _ in range(BATCH_SIZE)]
            # predicting_embed_inputs = np.asarray(predicting_embed_inputs)


            answer_logits = sess.run(model.predicting_logits,
                                     feed_dict={model.encoder_inputs.name: ids_inputs,
                                                model.encoder_inputs_length.name: inputs_length})

            prediction = recover(answer_logits.tolist()[0], index2char, False)
            # print(answer_logits.tolist()[0])
            print("predict->", prediction)
            print("-----------------------")


def metrics(data_path, dict_path, model_path):
    char2index, index2char = init_dict(dict_path)
    global VOCAB_SIZE
    VOCAB_SIZE = len(char2index)
    querys = []
    labels = []
    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip().replace(' ', '').encode('utf-8')
            label = line.split('#')[0]
            query = line.split('#')[1]
            querys.append(query)
            labels.append(label)

    correct = 0.0
    total = 0

    model = BeliefRnn(char2index, False)
    model.build_graph()

    saver = tf.train.Saver()
    # loaded_graph = tf.Graph()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver, model_path)
        for index, query in enumerate(querys):
            query = query.strip().decode('utf-8')
            ids = [char2index.get(c, 3) for c in query]
            inputs_length = [len(ids)] * BATCH_SIZE
            ids_inputs = [ids for _ in range(BATCH_SIZE)]
            answer_logits = sess.run(model.predicting_logits,
                                     feed_dict={model.encoder_inputs.name: ids_inputs,
                                                model.encoder_inputs_length.name: inputs_length})

            prediction = recover(answer_logits.tolist()[0], index2char, False)
            total += 1
            if prediction != labels[index]:
                print('{0}: {1}-->{2}'.format(query, labels[index], prediction))
            else:
                correct += 1
        print('accuracy:{0}'.format(correct / total))


def debug():
    data_path = '../data/sc/train/sale_train0830.txt'
    model_path = '../model/sc/belief_rnn/belief_rnn'
    encoder_vocab, embeddings = load_fasttext('../data/sc/train/belief_rnn/encoder_vocab.txt')
    decoder_vocab = load_decoder_vocab('../data/sc/train/belief_rnn/decoder_vocab.txt')

    gen = generate_batch_data(data_path, encoder_vocab=encoder_vocab, decoder_vocab=decoder_vocab)
    for a, b, c, d, e in gen:
        pass


def main():
    data_path = '../data/sc/train/sale_train0831.txt'
    model_path = '../model/sc/belief_rnn_0831/belief_rnn'
    char2index_path = '../data/sc/dict/char2index.txt'
    index2char_path = '../data/sc/dict/index2char.txt'
    dict_path = [char2index_path, index2char_path]
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', choices={'train', 'predict', 'valid'},
                        default='train', help='mode.if not specified,it is in train mode')
    args = parser.parse_args()

    if args.m == 'train':
        train(data_path, dict_path, model_path)
    elif args.m == 'predict':
        predict(dict_path, model_path)
    elif args.m == 'valid':
        metrics(data_path, dict_path, model_path)


if __name__ == '__main__':
    # debug()
    main()
    # test()
    # build_vocab('../data/sc/dict/big.txt','../data/sc/train/belief_rnn/char_vocab.txt')
