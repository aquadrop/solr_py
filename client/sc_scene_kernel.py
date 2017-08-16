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
        scene, sugg_scene, q = self.regex_plugin(q)
        if scene:
            return scene, sugg_scene, q
        try:
            if not self.web:
                if not self.clf:
                    return 'sale', None, q
                labels, _ = self.clf.predict(question=q)
                select = self.select_label(labels)
                ## qa plugin:
                if select == 'qa':
                    return select, 'sale', q
                if select == 'greeting':
                    return select, 'base', q
                return select, None, q
            else:
                text = requests.get('http://localhost:11305/sc/scene?q=' + q)
                return text.text, None, q
        except:
            return None, q

    base_pattern = re.compile(ur'.*?(天气|下雨吗|晴天吗|阴天吗|几点|呵呵|烦人|笨|蠢|滚粗|傻).*?')
    sing_pattern = re.compile(ur'.*?(((唱.*?(歌|曲)).*?)|((来|唱).*?(首).*?)|(我想听)|七里香|轨迹|星晴).*?')
    sale_pattern = re.compile(ur'.*?(买|吃|随便|看看).*?')
    qa_pattern = re.compile(ur'.*?((存|寄).*?包|在哪|在那|在几楼|在几层|怎么走|带我去|卫生间|厕所|积分|地铁|我要去|包装|停车场|电梯|出口|我想去|洗手间|充电|童车).*?')
    qa_clean_pattern = re.compile(ur'在哪里|在哪|在那里|在那|怎么走|带我去下|带我去')
    greeting_pattern = re.compile(ur".*?(在吗|在嘛|名字|几岁|多少岁).*?")
    greeting_clean_pattern = re.compile(ur'啊|呢|呀')
    """
    return direction, suggested_direction, fixed q
    """
    def regex_plugin(self, q):
        # q = QueryUtils.static_corenlp_cut(q, remove_tags=QueryUtils.remove_tags)
        try:
            if re.match(self.base_pattern, q):
                return 'base', None, q
            if re.match(self.qa_pattern, q):
                # q = re.sub(self.qa_clean_pattern, '', q)
                return 'qa', None, q
            if re.match(self.sing_pattern, q):
                return 'sing', None, q
            if re.match(self.sale_pattern, q):
                return 'sale', None, q
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
                return 'greeting', 'base', q
            return None, None, q
        except:
            return None, None, q

    def select_label(self, labels):
        """
        sale > qa > repeat > greeting = base
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
    print(re.match(SceneKernel.qa_pattern, u'欧米茄在哪里'))
    print(SK.kernel(u'七里香'))