#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask
from flask import request
import json

from kernel import Kernel
from gkernel import GKernel
from sequence_classifier import SeqClassifier
import business_qa_clf

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

app = Flask(__name__)

kernel = GKernel("../model/graph.pkl", "../model/seq_clf.pkl")


@app.route('/clf', methods=['GET', 'POST'])
def question_clf():
    pass


@app.route("/bot", methods=['GET', 'POST'])
def query():
    args = request.args
    q = args['q']
    print q
    return kernel.kernel(q)


@app.route("/chat", methods=['GET', 'POST'])
def chat():
    args = request.args
    q = args['q']
    print q
    r = kernel.kernel(q)
    result = {"question": q, "result": {"answer": r}, "user": "solr"}
    return json.dumps(result, ensure_ascii=False)


@app.route("/walk", methods=['GET', 'POST'])
def r_walk_with_pointer():
    msg = "normal"
    try:
        args = request.args
        q = args['q']
        try:
            s = args['s']
            slot, r = kernel.r_walk_with_pointer_with_clf(q, s.encode('utf8'))
        except Exception, e:
            slot, r = kernel.r_walk_with_pointer_with_clf(q, None)
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
