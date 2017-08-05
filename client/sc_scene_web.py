#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask
from flask import request
import json

from lru import LRU
from sc_scene_kernel import SceneKernel
from sc_scene_clf import SceneClassifier
from sc_multilabel_clf import Multilabel_Clf

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

app = Flask(__name__)

scene_kernel = SceneKernel()

@app.route('/sc/scene', methods=['GET', 'POST'])
def chat():
    try:
        args = request.args
        q = args['q']
        return scene_kernel.kernel(q)
    except:
        return 'sale'

if __name__ == "__main__":
    # SK = SceneKernel()
    # print(SK.kernel('你叫什么名字'))
    app.run(host='0.0.0.0', port=11305, threaded=True)
