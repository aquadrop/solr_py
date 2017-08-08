#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import re

import numpy as np

from sc_scene_clf import SceneClassifier
from query_util import QueryUtils

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

class SceneKernel:
    def __init__(self, web=False):
        self.web = web
        if not web:
            try:
                print('attaching scene kernel...')
                self.clf = SceneClassifier.get_instance('../model/sc/scene_clf.pkl')
            except Exception,e:
                print('failed to attach scene kernel..all inquires will be redirected to main kernel..', e.message)
        else:
            print('attaching scene web kernel...')

    def kernel(self, q):
        ## first try regex_plugin:
        scene, q = self.regex_plugin(q)
        if scene:
            return scene, q
        try:
            if not self.web:
                if not self.clf:
                    return 'sale'
                labels, _ = self.clf.predict(question=q)
                return self.select_label(labels), q
            else:
                text = requests.get('http://localhost:11305/sc/scene?q=' + q)
                return text.text, q
        except:
            return None, q

    qa_pattern = re.compile(ur'.*?(在哪|在那|在几楼|在几层|怎么走|带我去|卫生间|厕所|停车场|电梯|出口|我想去).*?')
    qa_clean_pattern = re.compile(ur'在哪里|在哪|在那里|在那|怎么走|带我去下|带我去')
    greeting_pattern = re.compile(ur".*?(在吗|在嘛|名字).*?")
    greeting_clean_pattern = re.compile(ur'啊|呢|呀')
    def regex_plugin(self, q):
        # q = QueryUtils.static_corenlp_cut(q, remove_tags=QueryUtils.remove_tags)
        try:
            if re.match(self.qa_pattern, q):
                q = re.sub(self.qa_clean_pattern, '', q)
                return 'qa', q
            if re.match(self.greeting_pattern, q):
                if (len(q)) > 1:
                    q = re.sub(self.greeting_clean_pattern, '', q)
                try:
                    q = QueryUtils.static_corenlp_cut(q, remove_tags=QueryUtils.remove_tags)
                    q = ''.join(q).decode('utf-8')
                    if not q:
                        q = u'你好'
                except:
                    pass
                if isinstance(q, str):
                    q = q.decode('unicode-escape').encode('utf-8')
                return 'greeting', q
            return None, q
        except:
            return None, q

    def select_label(self, labels):
        """
        sale > qa > repeat > greetings = base
        :param labels:
        :return:
        """

        if 'sale' in labels:
            return 'sale'
        else:
            if 'qa' in labels:
                return 'qa'
            else:
                if 'repeat_machine' in labels:
                    return 'repeat_machine'
                if 'repeat_user' in labels:
                    return 'repeat_user'
                else:
                    return np.random.choice(labels, 1)[0]


if __name__ == '__main__':
    SK = SceneKernel()
    # greeting_pattern = re.compile(ur'在吗|在嘛|名字')
    print(re.match(SceneKernel.qa_pattern, u'我问下卫生间在哪里'))
    print(SK.regex_plugin(u'你叫什么名字'))