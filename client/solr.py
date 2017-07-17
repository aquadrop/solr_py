#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask
from flask import request
import json

from kernel import Kernel
from gkernel import GKernel
from sequence_classifier import SeqClassifier
<<<<<<< HEAD
from business_qa_clf import BqClassifier
=======
from scene_clf import SceneClassifier
>>>>>>> e617172e1ab49744c1465ab087b8c2bf10394310

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

app = Flask(__name__)

kernel = GKernel("../model/graph.pkl", "../model/seq_clf.pkl")
q_clf = BqClassifier('../data/train_pruned_fixed.txt',
                     '../data/common_qa.txt', '../data/hudong.txt')
q_clf.bulid_ngram()

<<<<<<< HEAD

@app.route('/clf', methods=['GET', 'POST'])
=======
multi_l_kernels = LRU(200)
s_clf = SceneClassifier.get_instance('../model/scene/sceneclf.pkl')


@app.route('/scene', methods=['GET', 'POST'])
>>>>>>> e617172e1ab49744c1465ab087b8c2bf10394310
def question_clf():
    args = request.args
    q = args['q']
    print q
<<<<<<< HEAD
    label, prob = q_clf.interface(q)
=======
    label, prob = s_clf.interface(q)
>>>>>>> e617172e1ab49744c1465ab087b8c2bf10394310
    result = {'answer': label, 'prob': prob}
    return json.dumps(result, ensure_ascii=False)


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
