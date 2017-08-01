#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import _uniout


reload(sys)
sys.setdefaultencoding("utf-8")


def print_cn(q, others=''):
    print _uniout.unescape(str(q), 'utf8'), others


def cn(q):
    return _uniout.unescape(str(q), 'utf8')
