#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from datetime import datetime

from sc_belief_tracker import BeliefTracker
from sc_qa_kernel import QAKernel
from sc_greeting_kernel import GreetingKernel
from sc_base_kernel import BaseKernel
from sc_scene_kernel import SceneKernel
from sc_repeat_kernel import RepeatKernel
from sc_sing_kernel import SimpleSingKernel
from sc_reqmore_kernel import RequestMoreKernel
from cn_util import print_cn
from cn_util import print_out

from sc_belief_graph import BeliefGraph
from sc_belief_clf import Multilabel_Clf
from sc_scene_clf import SceneClassifier
from sc_qa_clf import SimpleSeqClassifier

class EntryKernel:
    ## static
    static_scene_kernel = None

    QA = 'qa'
    SALE = 'sale'
    GREETING = 'greeting'
    BASE = 'base'
    SING = 'sing'
    REQMORE = 'reqmore'

    def __init__(self):
        if not EntryKernel.static_scene_kernel:
            self.scene_kernel = SceneKernel(web=False)
            EntryKernel.static_scene_kernel = self.scene_kernel
        else:
            print('skipping attaching scene classifier as already attached...')
            self.scene_kernel = EntryKernel.static_scene_kernel
        # self.scene_kernel = SceneKernel(web=True)
        self.main_kernel = BeliefTracker("../model/sc/belief_graph.pkl", '../model/sc/belief_clf.pkl')
        self.qa_kernel = QAKernel()
        self.greeting_kernel = GreetingKernel()
        self.repeat_kernel = RepeatKernel()
        self.base_kernel = BaseKernel()
        self.sing_kernel = SimpleSingKernel()
        self.reqmore_kernel = RequestMoreKernel()
        self.last_response = None

    def kernel(self, q, direction=None, user='solr', debug=False, recursive=False):
        fixed_q = q
        suggested_direction = None
        if not direction:
            ## first determined by SceneKernel about directions
            direction, suggested_direction, fixed_q = self.scene_kernel.kernel(q)
            if not direction:
                return 'unable to respond as scene kernel is detached...'
            ## store value in repeat kernel
            self.repeat_kernel.store_user_q(q)

        q = fixed_q

        response = None
        inside_intentions = ''
        redirected = False
        if direction == EntryKernel.SING:
            response = self.sing_kernel.kernel(q)
        if direction == EntryKernel.BASE:
            response = self.base_kernel.kernel(q)
            if not response:
                response = '...'
        if direction == EntryKernel.QA:
            redirection, response = self.qa_kernel.kernel(q, self.last_response)
            if redirection:
                # response = self.kernel(q=q, direction=suggested_direction, debug=False, recursive=True)
                redirected = True
                if redirection == 'sale':
                    inside_intentions, response = self.main_kernel.kernel(query=q)
                if redirection == 'base':
                    response = self.base_kernel.kernel(query=q)
        if direction == EntryKernel.GREETING:
            direction, response = self.greeting_kernel.kernel(q)
            if not response and direction:
                redirected = True
                response = self.base_kernel.kernel(q)
        if direction == RepeatKernel.MACHINE:
            response = self.repeat_kernel.kernel(type_=RepeatKernel.MACHINE)
        if direction == RepeatKernel.USER:
            response = self.repeat_kernel.kernel(type_=RepeatKernel.USER)
        if direction == EntryKernel.SALE:
            inside_intentions, response = self.main_kernel.kernel(query=q)
            self.last_response = response
        if direction == EntryKernel.REQMORE:
            response = self.reqmore_kernel.kernel(q)

        if not response:
            suggested_direction = EntryKernel.BASE
            response = self.base_kernel.kernel(q)

        ## store response in repeat kernel:
        self.repeat_kernel.store_machine_q(r=response)
        current_date = time.strftime("%Y.%m.%d")
        log_file = '../logs/materials_' + current_date + '.log'
        current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        with open(log_file, 'a') as f:
            if not recursive:
                if suggested_direction and redirected:
                    if inside_intentions:
                        log = '用户:{0}##timestamp:{1}##问题:{2}##场景:{3}##修正场景:{4}##分类:{5}##答案:{6}'.format(user,\
                                                                                                     current_time,\
                                                                                           q, str(direction),\
                                                                                           str(suggested_direction),\
                                                                                           inside_intentions, response)
                    else:
                        log = '用户:{0}##timestamp:{1}##问题:{2}##场景:{3}##修正场景:{4}##答案:{5}'.format(user,\
                                                                                                     current_time, \
                                                                                           q, str(direction), \
                                                                                           str(suggested_direction), \
                                                                                           response)
                else:
                    if inside_intentions:
                        log = '用户:{0}##timestamp:{1}##问题:{2}##场景:{3}##分类:{4}##答案:{5}'.format(user,\
                                                                                                     current_time,\
                                                                                           q, str(direction),\
                                                                                           inside_intentions, response)
                    else:
                        log = '用户:{0}##timestamp:{1}##问题:{2}##场景:{3}##答案:{4}'.format(user,\
                                                                                                     current_time, \
                                                                                           q, str(direction), \
                                                                                           response)
                print_out(log, f)
        if debug:
            if suggested_direction and redirected:
                if inside_intentions:
                    return response + '@@scene_clf:' + str(direction) + '-->' + str(suggested_direction) + '@@belief_tracker:' + inside_intentions
                else:
                    return response + '@@scene_clf:' + str(direction) + '-->' + str(suggested_direction)
            else:
                if inside_intentions:
                    return response + '@@scene_clf:' + str(direction) + '@@belief_tracker:' + inside_intentions
                else:
                    return response + '@@scene_clf:' + str(direction)
        else:
            return response

def test(input_file, instance):
    with open(input_file, 'r') as f:
        current_user = None
        i = 0
        for line in f:
            try:
                i = i + 1
                components = line.split('##')
                user = components[0]
                if user != current_user:
                    current_user = user
                    instance.main_kernel.clear_state()
                    instance.last_response = None
                question = components[2]
                question = question.split(":")[1]
                answer = instance.kernel(question.decode('utf-8'))
                print('---%d--%s###%s' % (i, question, answer))
            except:
                instance.main_kernel.clear_state()

if __name__ == '__main__':
    kernel = EntryKernel()
    input_file = '../data/sc/test/test.txt'

    test(input_file, kernel)
    #
    while True:
        input_ = raw_input()
        input_ = input_.decode('utf-8')
        response = kernel.kernel(input_)
        print(response)
    #
    # response = kernel.kernel(u'我不买实惠的衣服')
    # print(response)
