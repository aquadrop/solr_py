#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import json
import argparse
from cn_util import print_cn
from itertools import chain

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

reload(sys)
sys.setdefaultencoding("utf-8")

import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np
from utils.mail import send_mail

ENC_VOCAB_SIZE = 3694
DEC_VOCAB_SIZE = 260
EMBEDDING_SIZE = 400
HIDDEN_UNIT = 256
N_LAYER = 10
BATCH_SIZE = 32
RNN_SIZE = 256
PAD = 2

char2index = None
index2char = None

label2index = None
index2label = None


def init_dict(dict_path):
    char2index_f = open(dict_path[0], 'r')
    index2char_f = open(dict_path[1], 'r')
    label2index_f = open(dict_path[2], 'r')
    index2label_f = open(dict_path[3], 'r')

    global char2index
    char2index = json.load(char2index_f)
    global index2char
    index2char = json.load(index2char_f)
    global label2index
    label2index = json.load(label2index_f)
    global index2label
    index2label = json.load(index2label_f)


def recover(index, source=True):
    sentence = []
    ignore = [0, 2, 3]
    for ii in index:
        if ii in ignore:
            continue
        if ii == 1:
            break
        if source:
            sentence.append(str(index2char[str(ii)]))
        else:
            sentence.append(str(index2label[str(ii)]))
    return ''.join(sentence)


def padding(inputs):
    batch_size = len(inputs)
    sequence_lengths = [len(seq) for seq in inputs]
    max_sequence_length = max(sequence_lengths)

    inputs_batch_major = np.ones(
        (batch_size, max_sequence_length), np.int32) * PAD
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i][j] = element

    return inputs_batch_major


def generate_batch_data(data_path, batch_size=32):
    sources = list()
    targets = list()

    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            target = line.split('#')[0].split(',')
            space = [' ' for _ in target]
            target = list(chain(*zip(target, space)))[:-1]
            source = line.split('#')[1]
            sources.append(source)
            targets.append(target)
            # print_cn(target)
    size = len(sources)

    while True:
        ss = np.random.randint(0, size, batch_size)

        encoder_inputs = [
            [char2index.get(word, 3) for word in sources[line_num].strip().decode('utf-8')] for line_num in
            ss]
        decoder_inputs = [
            [label2index.get('#GO#')] + [label2index.get(word.decode('utf-8'), 3) for word in targets[line_num]]
            for
            line_num in ss]
        labels = [
            [label2index.get(word.decode('utf-8'), 3) for word in targets[line_num]] + [label2index.get('#EOS#')]
            for
            line_num in ss]
        encoder_inputs_length = [len(line) for line in encoder_inputs]
        decoder_inputs_length = [len(line) for line in decoder_inputs]

        encoder_inputs = padding(encoder_inputs)
        decoder_inputs = padding(decoder_inputs)
        labels = padding(labels)

        yield np.asarray(encoder_inputs), np.asarray(decoder_inputs), np.asarray(labels), np.asarray(
            encoder_inputs_length), np.asarray(decoder_inputs_length)

        # yield encoder_inputs, decoder_inputs, labels, encoder_inputs_length, decoder_inputs_length


class BeliefRnn:
    def __init__(self, trainable=True):
        self.trainable = trainable

    def _create_placeholder(self):
        print('Create placeholders......')

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

    def _inference(self):
        print('Create inference......')

        def get_cell(size):
            cell = tf.contrib.rnn.LSTMCell(size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            return cell

        with tf.variable_scope("encoder") as scope:
            encoder_embeddings = tf.Variable(tf.random_uniform([ENC_VOCAB_SIZE, EMBEDDING_SIZE]))
            encoder_embed_input = tf.contrib.layers.embed_sequence(self.encoder_inputs, ENC_VOCAB_SIZE, EMBEDDING_SIZE)
            encoder_cell = tf.contrib.rnn.MultiRNNCell([get_cell(RNN_SIZE) for _ in range(N_LAYER)])
            encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_embed_input,
                                                              sequence_length=self.encoder_inputs_length,
                                                              dtype=tf.float32)


        with tf.variable_scope("decoder") as scope:
            decoder_embeddings = tf.Variable(tf.random_uniform([DEC_VOCAB_SIZE, EMBEDDING_SIZE]))
            decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, self.decoder_inputs)
            # decoder_embed_input = tf.contrib.layers.embed_sequence(self.decoder_inputs, DEC_VOCAB_SIZE, EMBEDDING_SIZE)
            decoder_cell = tf.contrib.rnn.MultiRNNCell([get_cell(RNN_SIZE) for _ in range(N_LAYER)])

            output_layer = Dense(DEC_VOCAB_SIZE,
                                 kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            self.max_target_sequence_length = tf.reduce_max(
                self.decoder_inputs_length, name='max_target_len')

        with tf.variable_scope("decoder") as scope:
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                                sequence_length=self.decoder_inputs_length,
                                                                time_major=False)

            training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                               helper=training_helper,
                                                               initial_state=encoder_state,
                                                               output_layer=output_layer)

            self.decoder_output, decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                                      impute_finished=True,
                                                                                      maximum_iterations=self.max_target_sequence_length)

        with tf.variable_scope("decoder", reuse=True):
            start_tokens = tf.tile(tf.constant([label2index['#GO#']], dtype=tf.int32), [BATCH_SIZE],
                                   name='start_tokens')
            predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(encoder_embeddings,
                                                                         start_tokens,
                                                                         label2index['#EOS#'])
            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                                 predicting_helper,
                                                                 encoder_state,
                                                                 output_layer)
            self.predicting_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                             impute_finished=True)
            self.predicting_logits = tf.identity(self.predicting_output.sample_id, name='predictions')

    def _create_loss(self):
        print('Create loss......')

        self.training_logits_ = tf.identity(self.decoder_output.rnn_output, 'logits')
        masks = tf.sequence_mask(self.decoder_inputs_length, self.max_target_sequence_length, dtype=tf.float32,
                                 name='masks')
        self.loss = tf.contrib.seq2seq.sequence_loss(self.training_logits_, self.labels_, masks)
        self.predictions_ = tf.argmax(self.training_logits_, axis=2)

    def _create_optimizer(self):
        print('Create optimizer......')

        self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    def build_graph(self):
        print('Build graph......')

        self._create_placeholder()
        self._inference()
        if self.trainable:
            self._create_loss()
            self._create_optimizer()


def _check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(
        os.path.dirname("../model/sc/belief_rnn/belief_rnn"))
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading parameters for the model")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Initializing fresh parameters for the model")


def _get_user_input():
    """ Get user's input, which will be transformed into encoder input later """
    print("query:->", end="")
    sys.stdout.flush()
    return sys.stdin.readline()


def train(data_path, dict_path):
    print('Training......')

    init_dict(dict_path)
    gen = generate_batch_data(data_path)
    model = BeliefRnn()
    model.build_graph()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)
        max_loss = 0.5
        i = 0
        for enci, deci, lab, encil, decil in gen:
            # print(enci)
            # print(deci)
            # print(lab)
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
                print('question:     >', recover(enci[0]))
                print('answer:       >', recover(deci[0], False))
                # print('prediction:   >', predictions[0])
                print('prediction:   >', recover(predictions[0], False))
                print('loss:         >', loss)

                if loss < max_loss:
                    max_loss = loss * 0.7
                    print('saving model...', i, loss)
                    saver.save(
                        sess, "../model/sc/belief_rnn/belief_rnn", global_step=i)
                if i % 10 == 0 and i > 1:
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
                            + 'question:     >' + recover(enci[0]) + '\n' \
                            + 'answer:       >' + recover(deci[0], False) + '\n' \
                            + 'prediction:   >' + recover(predictions[0], False) + '\n' \
                            + 'loss:         >' + str(loss)

                    send_mail(sends)
                    saver.save(
                        sess, "../model/sc/belief_rnn/belief_rnn", global_step=i)
            i = i + 1


def predict(dict_path):
    init_dict(dict_path)
    model = BeliefRnn(False)
    model.build_graph()

    saver = tf.train.Saver()
    # loaded_graph = tf.Graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)
        while True:
            line = _get_user_input()
            translation = [char2index.get(word, 3) for word in line.strip().decode('utf-8')]
            translation = [char2index.get('#GO#')] + translation
            encoder_input = np.tile([translation], (BATCH_SIZE, 1))
            encoder_lengths = np.array([len(o) for o in encoder_input])

            # print(encoder_input.shape)

            # logits = loaded_graph.get_tensor_by_name('predictions:0')

            answer_logits = sess.run(model.predicting_logits,
                                     feed_dict={model.encoder_inputs.name: encoder_input,
                                                model.encoder_inputs_length.name: encoder_lengths})

            prediction = recover(answer_logits.tolist()[0],False)
            # print_cn(prediction)
            print("predict->", prediction)
            print("-----------------------")

def interface(query):



def debug():
    dict_path = ['../data/sc/dict/char2index_dict_big.txt', '../data/sc/dict/index2char_dict_big.txt',
                 '../data/sc/dict/label2index.txt', '../data/sc/dict/index2label.txt']
    init_dict(dict_path)
    gen = generate_batch_data('../data/sc/train/sale_v2.txt')
    for enci, deci, lab, encil, decil in gen:
        print(enci)
        print(deci)
        print(lab)


def main():
    dict_path = ['../data/sc/dict/char2index_dict_big.txt', '../data/sc/dict/index2char_dict_big.txt',
                 '../data/sc/dict/label2index.txt', '../data/sc/dict/index2label.txt']
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', choices={'train', 'predict'},
                        default='train', help='mode.if not specified,it is in train mode')
    args = parser.parse_args()

    if args.m == 'train':
        train('../data/sc/train/sale_v2.txt', dict_path)
    elif args.m == 'predict':
        predict(dict_path)


if __name__ == '__main__':
    # debug()
    main()
