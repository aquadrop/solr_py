#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sc_main_kernel import SCKernel
from sc_qa_kernel import QAKernel
from sc_greeting_kernel import GreetingKernel
from sc_base_kernel import BaseKernel
from sc_scene_kernel import SceneKernel
from sc_repeat_kernel import RepeatKernel

from sc_scene_clf import SceneClassifier
from sc_multilabel_clf import Multilabel_Clf

class EntryKernel:
    ## static
    # scene_kernel = SceneKernel()

    QA = 'qa'
    SALE = 'sale'
    GREETING = 'greeting'
    BASE = 'base'

    def __init__(self):
        self.scene_kernel = SceneKernel(True)
        self.main_kernel = SCKernel("../model/sc_graph_v7.pkl", '../model/sc/multilabel_clf.pkl')
        self.qa_kernel = QAKernel()
        self.greeting_kernel = GreetingKernel()
        self.repeat_kernel = RepeatKernel()
        self.base_kernel = BaseKernel()

    def kernel(self, q, direction=None):
        if not direction:
            ## first determined by SceneKernel about directions
            direction, fixed_q = self.scene_kernel.kernel(q)
            if not direction:
                return 'unable to respond as scene kernel is detached...'
            ## store value in repeat kernel
            self.repeat_kernel.store_user_q(q)

        q = fixed_q

        response = None
        if direction == EntryKernel.BASE:
            response = self.base_kernel.kernel(q)
        if direction == EntryKernel.QA:
            response = self.qa_kernel.kernel(q)
        if direction == EntryKernel.GREETING:
            response = self.greeting_kernel.kernel(q)
            if not response:
                self.kernel(q, direction=EntryKernel.BASE)
        if direction == RepeatKernel.MACHINE:
            response = self.repeat_kernel.kernel(type_=RepeatKernel.MACHINE)
        if direction == RepeatKernel.USER:
            response = self.repeat_kernel.kernel(type_=RepeatKernel.USER)
        if direction == EntryKernel.SALE:
            _, response = self.main_kernel.kernel(query=q)

        if not response:
            self.kernel(q=q, direction=EntryKernel.BASE)

        ## store response in repeat kernel:
        self.repeat_kernel.store_machine_q(r=response)
        return response

if __name__ == '__main__':
    kernel = EntryKernel()
    print(kernel.kernel(u'停车场在哪'))