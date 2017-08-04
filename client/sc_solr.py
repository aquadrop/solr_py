#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Queue

from flask import Flask
from flask import request
import json

from lru import LRU
from sc_kernel import EntryKernel
from sc_scene_clf import SceneClassifier
from sc_multilabel_clf import Multilabel_Clf

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

app = Flask(__name__)

kernel = EntryKernel()
multi_sc_kernels = LRU(200)

QSIZE = 1
kernel_backups = Queue.Queue(QSIZE)

@app.route('/sc/chat', methods=['GET', 'POST'])
def chat():
    try:
        args = request.args
        q = args['q']
        q = q.decode('unicode-escape').encode('utf-8')
        try:
            u = args['u']
            if not multi_sc_kernels.has_key(u):
                if kernel_backups.qsize() > 0:
                    ek = kernel_backups.get_nowait()
                    multi_sc_kernels[u] = ek
                else:
                    for i in xrange(30):
                        k = EntryKernel()
                        kernel_backups.put_nowait(k)
                        print('========================')
            u_i_kernel = multi_sc_kernels[u]
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
    # SK = SceneKernel()
    # print(SK.kernel('你叫什么名字'))
    for i in xrange(QSIZE):
        print('========================')
        k = EntryKernel()
        kernel_backups.put_nowait(k)
    app.run(host='0.0.0.0', port=3000, threaded=True)
