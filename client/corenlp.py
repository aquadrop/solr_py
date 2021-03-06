# -*- coding:utf-8 -*-

from jpype import *
import os.path
import sys

from flask import Flask
from flask import request

import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import jieba

app = Flask(__name__)

reload(sys)
sys.setdefaultencoding('utf-8')


class CoreNLP:
    def __init__(self):
        self.jarpath = "./"
        self.path = os.path.join(os.path.abspath('../lib/'), self.jarpath)
        self.stanfordCoreNLP = None
        self.coreNLPService = None
        self.startJVM()

    def startJVM(self):
        startJVM(getDefaultJVMPath(), "-Djava.ext.dirs=%s" % self.path)
        StanfordCoreNLP = JClass("edu.stanford.nlp.pipeline.StanfordCoreNLP")
        CoreNLPService = JClass("edu.stanford.nlp.pipeline.demo.CoreNLPService")
        self.stanfordCoreNLP = StanfordCoreNLP("edu/stanford/nlp/pipeline/StanfordCoreNLP-chinese.properties")
        self.coreNLPService = CoreNLPService()

    def segment(self, sent):
        segmentation = self.coreNLPService.segment(self.stanfordCoreNLP, sent)
        return segmentation

    def pos(self, sent):
        tagging = self.coreNLPService.pos(self.stanfordCoreNLP, sent)
        return tagging

    def shutdownJVM(self):
        shutdownJVM()

# if __name__ == "__main__":
#     coreNLP = CoreNLP()
#     while True:
#         sent = raw_input("-->")
#         segmentation = coreNLP.segment(sent)
#         tagging = coreNLP.pos(sent)
#         print segmentation, tagging

coreNLP = CoreNLP()

@app.route("/tokenize",methods=['GET', 'POST'])
def query():
    args = request.args
    q = args['q']
    return coreNLP.pos(q)

if __name__ == "__main__":
    app.run()
