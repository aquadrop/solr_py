#!/usr/bin/env python
# -*- coding: utf-8 -*-
import bisect
from node import Node

class RangeSearchNode(Node):

    ## define edges
    value_type = "RANGE" ## key, range, date...

    breakpoints = []
    # mainly use value to as dictionary key
    # might use breakpoint, the query time is log(N)

    def __init__(self, slot, value, breakpoints):
        self.slot = slot
        self.value = value
        self.breakpoints = breakpoints

    def decide(self, q):
        return self.grade(float(q), self.neighbors)

    @staticmethod
    def grade(self, score, breakpoints, nodes):
        i = bisect(breakpoints, score)
        return nodes[i]
    # def grade(score, breakpoints=[60, 70, 80, 90], grades='FDCBA'):
    #     i = bisect(breakpoints, score)
    #     return grades[i]