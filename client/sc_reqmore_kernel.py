#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

import numpy as np

from cn_util import print_cn

class RequestMoreKernel:
    temporary = ["心好累懒得帮你找了","这些都挺好的别挑了","您喜欢什么啊"]
    def __init__(self):
        pass

    def kernel(self, q):
        return np.random.choice(self.temporary, 1)[0]