#!/usr/bin/env python
# -*- coding: utf-8 -*-

class RepeatKernel:
    MACHINE = 'repeat_machine'
    USER = 'repeat_user'

    def __init__(self):
        print('attaching repeat kernel...')
        self.r = '~~~'
        self.q = '~~~'

    def kernel(self, type_):
        if type_ == RepeatKernel.MACHINE:
            return self.r
        if type_ == RepeatKernel.USER:
            return self.q

    def store_machine_q(self, r):
        self.r = r

    def store_user_q(self, q):
        self.q = q