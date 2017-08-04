#!/usr/bin/env python
# -*- coding: utf-8 -*-

from client.sc.sckernel import SCKernel
from client.sc.qa_kernel import QAKernel
from client.sc.greeting_kernel import GreetingKernel
from client.sc.base_kernel import BaseKernel
from client.sc.scene_kernel import SceneKernel
from client.sc.repeat_kernel import RepeatKernel

from client.sc.multilabel_clf import Multilabel_Clf
from client.sc.scene_clf import SceneClassifier
from cn_util import print_cn
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

    def kernel(self, q, direction=None):
        if not direction:
            ## first determined by SceneKernel about directions
            direction = Kernel.scene_kernel.kernel(q)
            print_cn('direction:',direction)
            ## store value in repeat kernel
            self.repeat_kernel.store_user_q(q)

        response = None
        if direction == Kernel.BASE:
            response = self.base_kernel.kernel(q)
        if direction == Kernel.QA:
            response = self.qa_kernel.kernel(q)
        if direction == Kernel.GREETING:
            response = self.greeting_kernel.kernel(q)
            if not response:
                self.kernel(q, direction=Kernel.BASE)
        if direction == RepeatKernel.MACHINE:
            response = self.repeat_kernel.kernel(type_=RepeatKernel.MACHINE)
        if direction == RepeatKernel.USER:
            response = self.repeat_kernel.kernel(type_=RepeatKernel.USER)
        if direction == Kernel.SALE:
            _, response = self.main_kernel.kernel(query=q)

        if not response:
            self.kernel(q=q, direction=Kernel.BASE)

        ## store response in repeat kernel:
        self.repeat_kernel.store_machine_q(r=response)
        return response

if __name__ == '__main__':
    kernel = Kernel()
    print(kernel.kernel('购物'))