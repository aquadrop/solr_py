#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Queue
import argparse
import traceback

from urllib import unquote
from flask import Flask
from flask import request
import json

from lru import LRU
from sc_entry_kernel import EntryKernel
from sc_belief_graph import BeliefGraph
from sc_belief_clf import Multilabel_Clf
from sc_scene_clf import SceneClassifier
from sc_qa_clf import SimpleSeqClassifier

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

app = Flask(__name__)

kernel = EntryKernel()
multi_sc_kernels = LRU(1000)

QSIZE = 1
kernel_backups = Queue.Queue(1000)

@app.route('/sc/info', methods=['GET', 'POST'])
def info():
    current_q_size = kernel_backups.qsize()
    current_u_size = len(multi_sc_kernels.keys())
    result = {"current_q_size": current_q_size, "current_u_size": current_u_size}
    return json.dumps(result, ensure_ascii=False)

@app.route('/sc/chat', methods=['GET', 'POST'])
def chat():
    try:
        args = request.args
        debug = True
        try:
            debug = bool(args['debug'])
        except:
            pass
        q = args['q']
        q = unquote(q)
        if isinstance(q, str):
            q = q.decode('unicode-escape').decode('utf-8')
        try:
            u = args['u']
            if not multi_sc_kernels.has_key(u):
                if kernel_backups.qsize() > 0:
                    ek = kernel_backups.get_nowait()
                    multi_sc_kernels[u] = ek
                else:
                    for i in xrange(2):
                        k = EntryKernel()
                        kernel_backups.put_nowait(k)
                        result = {"question": q, "result": \
                            {"answer": "主机负载到达初始上限,正在为您分配实例..."},
                                  "user": u}
                        # print('========================')
                    return json.dumps(result, ensure_ascii=False)
            u_i_kernel = multi_sc_kernels[u]
            r = u_i_kernel.kernel(q=q, debug=debug, user=u)
            result = {"question": q, "result": {"answer": r}, "user": u}
            return json.dumps(result, ensure_ascii=False)

        except:
            traceback.print_exc()
            r = kernel.kernel(q=q, debug=debug)
            result = {"question": q, "result": {"answer": r}, "user": "solr"}
            return json.dumps(result, ensure_ascii=False)
    except Exception, e:
        traceback.print_exc()
        result = {"question": q, "result": {"answer": "主机负载到达上限或者主机核心出现异常"}, "user": "solr"}
        return json.dumps(result, ensure_ascii=False)

if __name__ == "__main__":
    # SK = SceneKernel()
    # print(SK.kernel('你叫什么名字'))

    parser = argparse.ArgumentParser()
    parser.add_argument('--qsize', choices={'1', '8', '200'},
                        default='200', help='q_size initializes number of the starting instances...')
    args = parser.parse_args()

    QSIZE = int(args.qsize)

    for i in xrange(QSIZE):
        print('========================')
        k = EntryKernel()
        kernel_backups.put_nowait(k)
    app.run(host='0.0.0.0', port=11304, threaded=True)
