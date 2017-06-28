#!/usr/bin/env python
# -*- coding: utf-8 -*-
from node import Node

class KeySearchNode(Node):

    ## define edges
    value_type = "KEY" ## key, range, date...

    def __init__(self, slot, value):
        self.value = value
        self.slot = slot