#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sckernel import SCKernel
from qa_kernel import QAKernel
from greeting_kernel import GreetingKernel
from base_kernel import BaseKernel
from scene_kernel import SceneKernel
from repeat_kernel import RepeatKernel

from multilabel_clf import Multilabel_Clf

class Kernel:
    ## static
    scene_kernel = SceneKernel()

    QA = 'qa'
    SALE = 'sale'
    GREETING = 'greeting'
    BASE = 'base'

    def __init__(self):
        self.main_kernel = SCKernel("../../model/sc_graph_v7.pkl", '../../model/sc/multilabel_clf.pkl')
        self.qa_kernel = QAKernel()
        self.greeting_kernel = GreetingKernel()
        self.repeat_kernel = RepeatKernel()
        self.base_kernel = BaseKernel()

    def kernel(self, q):
        ## first determined by SceneKernel about directions
        direction = Kernel.scene_kernel.kernel(q)
        ## store value in repeat kernel
        self.repeat_kernel.store_user_q(q)

        if direction == Kernel.BASE:
            response = self.base_kernel.kernel(q)
        if direction == Kernel.QA:
            response = self.qa_kernel.kernel(q)
        if direction == Kernel.GREETING:
            response = self.greeting_kernel(q)
        if direction == Kernel.BASE:
            response = self.base_kernel.kernel(q)
        if direction == RepeatKernel.MACHINE:
            response = self.repeat_kernel.kernel(type_=RepeatKernel.MACHINE)
        if direction == RepeatKernel.USER:
            response = self.repeat_kernel.kernel(type_=RepeatKernel.USER)
        if direction == Kernel.SALE:
            _, response = self.main_kernel.kernel(query=q)

        ## store response in repeat kernel:
        self.repeat_kernel.store_machine_q(r=response)
        return response

if __name__ == '__main__':
    kernel = Kernel()
    print(kernel.kernel('购物'))