#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import numpy as np
import json
import client.cn_util
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def test():
    url = "http://localhost:11304/sc/chat?q=%s&u=%s"

    while True:
        qn = [u'吃饭']
        u = np.random.randint(0, 10)
        q = np.random.choice(qn, 1)[0]
        r_url = url % (q, str(u))
        r = requests.get(r_url)
        text = json.dumps(r.json(), ensure_ascii=False)
        client.cn_util.print_cn(text)
        if 'unclear' in text or 'base' in text:
            print('---------')

if __name__ == '__main__':
    test()