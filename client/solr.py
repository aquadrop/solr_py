#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask
from kernel import Kernel
from gkernel import GKernel
from flask import request
import json
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

app = Flask(__name__)

kernel = GKernel("../model/graph.pkl")

@app.route("/bot",methods=['GET', 'POST'])
def query():
    args = request.args
    q = args['q']
    print q
    return kernel.kernel(q)

@app.route("/chat",methods=['GET', 'POST'])
def chat():
    args = request.args
    q = args['q']
    print q
    r = kernel.kernel(q)
    result = {"question":q, "result":{"answer":r}, "user":"solr"}
    return json.dumps(result, ensure_ascii=False)

@app.route("/walk",methods=['GET', 'POST'])
def r_walk_with_pointer():
    args = request.args
    q = args['q']
    s = args['s']
    print q, s
    slot, r = kernel.r_walk_with_pointer(q, s.encode('utf8'))
    result = {"answer":r, "slot":slot}
    return json.dumps(result, ensure_ascii=False)

@app.route("/clear",methods=['GET', 'POST'])
def clear():
    kernel.clear_state()
    return "state cleared"

if __name__ == "__main__":
    app.run(port=11303, threaded=True)
