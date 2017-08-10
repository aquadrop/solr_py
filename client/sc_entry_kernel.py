#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sc_belief_tracker import BeliefTracker
from sc_qa_kernel import QAKernel
from sc_greeting_kernel import GreetingKernel
from sc_base_kernel import BaseKernel
from sc_scene_kernel import SceneKernel
from sc_repeat_kernel import RepeatKernel
from sc_sing_kernel import SimpleSingKernel
from cn_util import print_cn

from sc_belief_graph import BeliefGraph
from sc_belief_clf import Multilabel_Clf
from sc_scene_clf import SceneClassifier

class EntryKernel:
    ## static
    static_scene_kernel = None

    QA = 'qa'
    SALE = 'sale'
    GREETING = 'greeting'
    BASE = 'base'
    SING = 'sing'

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

    def kernel(self, q, direction=None, debug=False, recursive=False):
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
        if direction == EntryKernel.SING:
            response = self.sing_kernel.kernel(q)
        if direction == EntryKernel.BASE:
            response = self.base_kernel.kernel(q)
            if not response:
                response = '...'
        if direction == EntryKernel.QA:
            sucess, response = self.qa_kernel.kernel(q)
            if not sucess and suggested_direction:
                # response = self.kernel(q=q, direction=suggested_direction, debug=False, recursive=True)
                inside_intentions, response = self.main_kernel.kernel(query=q)
        if direction == EntryKernel.GREETING:
            suc, response = self.greeting_kernel.kernel(q)
            if not sucess and suggested_direction:
                response = self.base_kernel.kernel(q)
        if direction == RepeatKernel.MACHINE:
            response = self.repeat_kernel.kernel(type_=RepeatKernel.MACHINE)
        if direction == RepeatKernel.USER:
            response = self.repeat_kernel.kernel(type_=RepeatKernel.USER)
        if direction == EntryKernel.SALE:
            inside_intentions, response = self.main_kernel.kernel(query=q)

        if not response:
            suggested_direction = EntryKernel.BASE
            response = self.base_kernel.kernel(q)

        ## store response in repeat kernel:
        self.repeat_kernel.store_machine_q(r=response)
        if not recursive:
            print_cn('问题：{0}, 场景：{1}, 建议场景:{2}, 分类:{3}, 答案：{4}'.format(q, str(direction), str(suggested_direction), inside_intentions, response))
        if debug:
            if suggested_direction:
                if inside_intentions:
                    return response + '@@scene_clf:' + str(direction) + '->' + str(suggested_direction) + '@@belief_tracker:' + inside_intentions
                else:
                    return response + '@@scene_clf:' + str(direction) + '->' + str(suggested_direction)
            else:
                if inside_intentions:
                    return response + '@@scene_clf:' + str(direction) + '@@belief_tracker:' + inside_intentions
                else:
                    return response + '@@scene_clf:' + str(direction)
        else:
            return response


if __name__ == '__main__':
    kernel = EntryKernel()
    response = kernel.kernel(u'清淡的')
    print(response)
