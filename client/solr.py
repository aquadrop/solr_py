#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask
from flask import request
import json

from lru import LRU

from kernel import Kernel
from gkernel import GKernel
from qa_kernel import QAKernel
from interactive_kernel import IKernel
from sequence_classifier import SeqClassifier
from scene_clf import SceneClassifier

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

app = Flask(__name__)

kernel = GKernel("../model/graph_v7.pkl", "../model/seq_clf_v7.pkl")
qa_kernel = QAKernel()
i_kernel = IKernel()

multi_l_kernels = LRU(200)
s_clf = SceneClassifier.get_instance('../model/scene/sceneclf_six.pkl')


@app.route('/scene', methods=['GET', 'POST'])
def question_clf():
    args = request.args
    q = args['q']
    print q
    label, prob = s_clf.interface(q)
    result = {'cls': label, 'distribution': prob}
    return json.dumps(result, ensure_ascii=False)


@app.route("/bot", methods=['GET', 'POST'])
def query():
    args = request.args
    q = args['q']
    print q
    return kernel.kernel(q)


@app.route("/base", methods=['GET', 'POST'])
def chat():
    args = request.args
    q = args['q']
    print q
    r = kernel.kernel(q)
    result = {"question": q, "result": {"answer": r}, "user": "solr"}
    return json.dumps(result, ensure_ascii=False)


@app.route("/qa", methods=['GET', 'POST'])
def qa():
    try:
        args = request.args
        q = args['q']
        r = qa_kernel.kernel(q)
        result = {"question": q, "result": {"answer": r}, "user": "solr"}
        return json.dumps(result, ensure_ascii=False)
    except Exception, e:
        return json.dumps({"msg": e.message})


@app.route("/interactive", methods=['GET', 'POST'])
def interactive():
    try:
        args = request.args
        q = args['q']
        try:
            u = args['u']
            if not multi_l_kernels.has_key(u):
                multi_l_kernels[u] = IKernel()
            u_i_kernel = multi_l_kernels[u]
            r = u_i_kernel.kernel(q)
            result = {"question": q, "result": {"answer": r}, "user": "solr"}
            return json.dumps(result, ensure_ascii=False)

        except:
            r = i_kernel.kernel(q)
            result = {"question": q, "result": {"answer": r}, "user": "solr"}
            return json.dumps(result, ensure_ascii=False)
    except Exception, e:
        return json.dumps({"msg": e.message})


@app.route("/walk", methods=['GET', 'POST'])
def r_walk_with_pointer():
    msg = "normal"
    try:
        args = request.args
        q = args['q']
        try:
            s = args['s']
            slot, r = kernel.r_walk_with_pointer_with_clf(
                q.encode('utf-8'), s.encode('utf8'))
        except Exception, e:
            slot, r = kernel.r_walk_with_pointer_with_clf(
                q.encode('utf-8'), None)
    except Exception, e:
        kernel.clear_state()
        slot = None
        r = None
        msg = e.message

    result = {"answer": r, "slot": slot, "msg": msg}
    return json.dumps(result, ensure_ascii=False)


@app.route("/clear", methods=['GET', 'POST'])
def clear():
    kernel.clear_state()
    return "state cleared"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=11303, threaded=True)
