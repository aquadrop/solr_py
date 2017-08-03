#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask
from flask import request
import json

from lru import LRU

from multilabel_clf import Multilabel_Clf
from kernel import Kernel

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

app = Flask(__name__)

kernel = Kernel()
multi_l_kernels = LRU(200)

@app.route('/sc/chat', methods=['GET', 'POST'])
def chat():
    try:
        args = request.args
        q = args['q']
        try:
            u = args['u']
            if not multi_l_kernels.has_key(u):
                multi_l_kernels[u] = Kernel()
            u_i_kernel = multi_l_kernels[u]
            r = u_i_kernel.kernel(q)
            result = {"question": q, "result": {"answer": r}, "user": u}
            return json.dumps(result, ensure_ascii=False)

        except:
            r = kernel.kernel(q)
            result = {"question": q, "result": {"answer": r}, "user": "solr"}
            return json.dumps(result, ensure_ascii=False)
    except Exception, e:
        return json.dumps({"msg": e.message})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=11303, threaded=True)
