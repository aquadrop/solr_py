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
import gensim
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

ENC_VOCAB_SIZE = 3694
DEC_VOCAB_SIZE = 260
EMBEDDING_SIZE = 300
HIDDEN_UNIT = 128
N_LAYER = 2
BATCH_SIZE = 32
RNN_SIZE = 128
PAD = 2
WORD2VEC_MODEL=None

def init_word2vec():
    print('Init word2vec model.')
    path='../model/word2vec/w2v_cb'
    global WORD2VEC_MODEL
    WORD2VEC_MODEL=gensim.models.Word2Vec.load(path)

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

def word2vec_wv(word):
    global WORD2VEC_MODEL
    if WORD2VEC_MODEL.__contains__(word.strip()):
        vector = WORD2VEC_MODEL.__getitem__(word.strip())
    else:
        vector = WORD2VEC_MODEL.__getitem__('unk')
        print('unk')
    result = [v for v in vector]
    return result

def recover_source(index, vocab):
    sentence = []
    ignore = ['#PAD#', '#UNK#', '#GO#']
    for ii in index:
        word = vocab[ii]
        if word in ignore:
            continue
        if ii == '#EOS#':
            break
        sentence.append(word)
    return ''.join(sentence)


def recover_label(index, vocab):
    sentence = []
    ignore = ['#PAD#', '#UNK#', '#GO#']
    for ii in index:
        word = vocab[ii]
        if word in ignore:
            continue
        if word == '#EOS#':
            break
        sentence.append(word)
    return ''.join(sentence)

def load_fasttext(encoder_vocab_path):
    print('Load encoder vocab, embedding by fasttext.')
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

def load_word2vec(encoder_vocab_path):
    print('Load encoder vocab, embedding by word2vec.')
    encoder_vocab=[]
    embeddings=[]

    with open(encoder_vocab_path) as f:
        for word in f:
            word=word.strip().replace(' ','')
            if word not in encoder_vocab:
                encoder_vocab.append(word)
                embedding=word2vec_wv(word)
                embeddings.append(embedding)
    return encoder_vocab,embeddings

def load_decoder_vocab(decoder_vocab_path):
    print('Load decoder vocab')
    decoder_vocab = []

    with open(decoder_vocab_path) as f:
        for word in f:
            word = word.strip().replace(' ', '')
            if word not in decoder_vocab:
                decoder_vocab.append(word)
    print('Done.')

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

class BeliefRnn:
    def __init__(self, decoder_vocab, trainable=True):
        self.trainable = trainable
        self.decoder_vocab = decoder_vocab

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

        with tf.variable_scope('embedding') as scope:
            self.embeddings = tf.Variable(tf.constant(0.0, shape=[ENC_VOCAB_SIZE, EMBEDDING_SIZE]),
                                          trainable=False, name="embeddings")
            self.embedding_placeholder = tf.placeholder(tf.float32, [ENC_VOCAB_SIZE, EMBEDDING_SIZE])
            self.embedding_init = self.embeddings.assign(self.embedding_placeholder)

        self.encoder_embed_inputs = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)
        self.decoder_embed_inputs = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)

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
        with tf.variable_scope("decoder") as scope:
            decoder_cell = tf.contrib.rnn.MultiRNNCell([get_cell(RNN_SIZE) for _ in range(N_LAYER)])
            output_layer = Dense(DEC_VOCAB_SIZE,
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

        with tf.variable_scope("decoder", reuse=True):
            greedy_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embeddings,
                                                                     start_tokens=tf.fill([BATCH_SIZE],
                                                                                          self.decoder_vocab.index(
                                                                                              '#GO#')),
                                                                     end_token=self.decoder_vocab.index('#EOS#'))
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


def _get_user_input():
    """ Get user's input, which will be transformed into encoder input later """
    print("query:->", end="")
    sys.stdout.flush()
    return sys.stdin.readline()


def train(data_path, encoder_vocab_path, decoder_vocab_path, model_path,embedding='word2vec'):
    print('Training...')

    if embedding=='word2vec':
        init_word2vec()
        encoder_vocab, embeddings = load_word2vec(encoder_vocab_path)
        global WORD2VEC_MODEL
        del WORD2VEC_MODEL
    else:
        encoder_vocab, embeddings = load_fasttext(encoder_vocab_path)
    decoder_vocab = load_decoder_vocab(decoder_vocab_path)
    global ENC_VOCAB_SIZE
    ENC_VOCAB_SIZE = len(encoder_vocab)
    global DEC_VOCAB_SIZE
    DEC_VOCAB_SIZE = len(decoder_vocab)
    global EMBEDDING_SIZE
    EMBEDDING_SIZE = len(embeddings[0])

    gen = generate_batch_data(data_path, encoder_vocab=encoder_vocab, decoder_vocab=decoder_vocab)
    model = BeliefRnn(decoder_vocab)
    model.build_graph()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(model.embedding_init, feed_dict={model.embedding_placeholder: embeddings})
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
                print('question:     >', recover_source(enci[0], encoder_vocab))
                print('answer:       >', recover_label(deci[0], decoder_vocab))
                # print('prediction:   >', predictions[0])
                print('prediction:   >', recover_label(predictions[0], decoder_vocab))
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
                            + 'question:     >' + recover_source(enci[0], encoder_vocab) + '\n' \
                            + 'answer:       >' + recover_label(deci[0], decoder_vocab) + '\n' \
                            + 'prediction:   >' + recover_label(predictions[0], decoder_vocab) + '\n' \
                            + 'loss:         >' + str(loss)

                    send_mail(sends)
                    saver.save(
                        sess, model_path, global_step=i)
            i = i + 1


def predict(data_path, encoder_vocab_path, decoder_vocab_path, model_path,embedding='word2vec'):
    if embedding=='word2vec':
        encoder_vocab, embeddings = load_word2vec(encoder_vocab_path)
    else:
        encoder_vocab, embeddings = load_fasttext(encoder_vocab_path)
    decoder_vocab = load_decoder_vocab(decoder_vocab_path)
    global ENC_VOCAB_SIZE
    ENC_VOCAB_SIZE = len(encoder_vocab)
    global DEC_VOCAB_SIZE
    DEC_VOCAB_SIZE = len(decoder_vocab)
    global EMBEDDING_SIZE
    EMBEDDING_SIZE = len(embeddings[0])

    for word in encoder_vocab:
        jieba.add_word(word)
    for word in decoder_vocab:
        jieba.add_word(word)

    model = BeliefRnn(decoder_vocab, False)
    model.build_graph()

    saver = tf.train.Saver()
    # loaded_graph = tf.Graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(model.embedding_init, feed_dict={model.embedding_placeholder: embeddings})
        _check_restore_parameters(sess, saver, model_path)
        while True:
            line = _get_user_input()
            predicting_inputs = list(jieba.cut(line.strip().decode('utf-8')))
            ids = []
            for word in predicting_inputs:
                if word in encoder_vocab:
                    ids.append(encoder_vocab.index(word))
                else:
                    ids.append(encoder_vocab.index('#UNK#'))
            print_cn(predicting_inputs)
            predicting_inputs_length = [len(predicting_inputs)] * BATCH_SIZE
            ids_inputs = [ids for _ in range(BATCH_SIZE)]
            # predicting_embed_inputs = [[fasttext_wv(word) for word in predicting_inputs] for _ in range(BATCH_SIZE)]
            # predicting_embed_inputs = np.asarray(predicting_embed_inputs)


            answer_logits = sess.run(model.predicting_logits,
                                     feed_dict={model.encoder_inputs.name: ids_inputs,
                                                model.encoder_inputs_length.name: predicting_inputs_length})

            prediction = recover_label(answer_logits.tolist()[0], decoder_vocab)
            # print(answer_logits.tolist()[0])
            print("predict->", prediction)
            print("-----------------------")

def metrics(data_path, encoder_vocab_path, decoder_vocab_path, model_path,embedding='word2vec'):
    if embedding=='word2vec':
        encoder_vocab, embeddings = load_word2vec(encoder_vocab_path)
    else:
        encoder_vocab, embeddings = load_fasttext(encoder_vocab_path)
    decoder_vocab = load_decoder_vocab(decoder_vocab_path)
    global ENC_VOCAB_SIZE
    ENC_VOCAB_SIZE = len(encoder_vocab)
    global DEC_VOCAB_SIZE
    DEC_VOCAB_SIZE = len(decoder_vocab)
    global EMBEDDING_SIZE
    EMBEDDING_SIZE = len(embeddings[0])

    for word in encoder_vocab:
        jieba.add_word(word)
    for word in decoder_vocab:
        jieba.add_word(word)

    querys=[]
    labels=[]
    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip().replace(' ', '').encode('utf-8')
            label = line.split('#')[0].split(',')
            query = line.split('#')[1]
            querys.append(query)
            labels.append(label)

    correct = 0.0
    total = 0

    model = BeliefRnn(decoder_vocab, False)
    model.build_graph()

    saver = tf.train.Saver()
    # loaded_graph = tf.Graph()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(model.embedding_init, feed_dict={model.embedding_placeholder: embeddings})
        _check_restore_parameters(sess, saver, model_path)
        for index, query in enumerate(querys):
            predicting_inputs=cut(query)
            ids = []
            for word in predicting_inputs:
                if word in encoder_vocab:
                    ids.append(encoder_vocab.index(word))
                else:
                    ids.append(encoder_vocab.index('#UNK#'))
            predicting_inputs_length = [len(predicting_inputs)] * BATCH_SIZE
            ids_inputs = [ids for _ in range(BATCH_SIZE)]
            answer_logits = sess.run(model.predicting_logits,
                                     feed_dict={model.encoder_inputs.name: ids_inputs,
                                                model.encoder_inputs_length.name: predicting_inputs_length})

            prediction = recover_label(answer_logits.tolist()[0], decoder_vocab)
            predictions=prediction.split('/')
            total += 1
            if set(predictions)!=set(labels[index]):
                print('{0}: {1}-->{2}'.format(query, ' '.join(labels[index]), ' '.join(predictions)))
            else:
                correct+=1
        print('accuracy:{0}'.format(correct / total))

def debug():
    data_path = '../data/sc/train/sale_train0824.txt'
    model_path = '../model/sc/belief_rnn/belief_rnn'
    encoder_vocab, embeddings = load_fasttext('../data/sc/train/belief_rnn/encoder_vocab.txt')
    decoder_vocab = load_decoder_vocab('../data/sc/train/belief_rnn/decoder_vocab.txt')

    gen = generate_batch_data(data_path, encoder_vocab=encoder_vocab, decoder_vocab=decoder_vocab)
    for a, b, c, d, e in gen:
        pass


def main():
    data_path = '../data/sc/train/sale_train0824.txt'
    encoder_vocab_path = '../data/sc/train/belief_rnn/encoder_vocab.txt'
    decoder_vocab_path = '../data/sc/train/belief_rnn/decoder_vocab.txt'
    model_path = '../model/sc/belief_rnn_0831/belief_rnn'
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', choices={'train', 'predict', 'valid'},
                        default='train', help='mode. if not specified,it is in train mode')

    args = parser.parse_args()

    if args.m == 'train':
        train(data_path, encoder_vocab_path, decoder_vocab_path, model_path)
    elif args.m == 'predict':
        predict(data_path, encoder_vocab_path, decoder_vocab_path, model_path)
    elif args.m == 'valid':
        metrics(data_path, encoder_vocab_path, decoder_vocab_path, model_path)


if __name__ == '__main__':
    # debug()
    main()
    # test()
