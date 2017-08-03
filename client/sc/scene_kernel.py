#!/usr/bin/env python
# -*- coding: utf-8 -*-

class SceneKernel:
    def __init__(self):
        try:
            print('attaching scene kernel...')
        except:
            print('failed to attach scene kernel..all inquires will be redirected to main kernel..')

    def kernel(self, q):
        return 'sale'