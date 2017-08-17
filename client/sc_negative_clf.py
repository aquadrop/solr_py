#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from cn_util import print_cn



negative_pattern = re.compile(ur'.*?(不要|不买|不是|不吃|不太|不喜欢|不爱|吃不起|买不起|不想).*?')

class Negative_Clf:
    def __init__(self):
        pass

    def predict(self,input_):
        try:
            input_=input_.decode('utf-8')
            if re.match(negative_pattern, input_):
                return re.sub(negative_pattern, '', input_), True
            else:
                return input_, False
        except:
            return False

