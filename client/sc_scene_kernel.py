#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import re
import traceback

import numpy as np

from sc_scene_clf_fasttext import SceneClassifier
from query_util import QueryUtils
from solr_utils import SolrUtils
from sc_negative_clf import Negative_Clf
import cn_util

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

class SceneKernel:
    def __init__(self, web=False):
        self.web = web
        self.neg = Negative_Clf()
        if not web:
            try:
                print('attaching scene kernel...')
                self.clf = SceneClassifier.get_instance('../model/sc/scene_embeded_clf.pkl')
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
                q = QueryUtils.static_remove_pu(q)
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

    base_pattern = re.compile(ur'.*?(天气|下雨吗|晴天吗|阴天吗|几点|呵呵|烦人|笨|蠢|滚粗|傻|小德|请我吃).*?')
    sing_pattern = re.compile(ur'.*?(((唱.*?(歌|曲)).*?)|((来|唱).*?(首).*?)|(我想听)|七里香|轨迹|星晴).*?')
    sing_diff_pattern = re.compile(ur'.*?(我们.*?唱|我.*?唱).*?')
    sale_pattern = re.compile(ur'.*?(买|吃|随便|看看|饿).*?')
    qa_pattern = re.compile(ur'.*?((存|寄).*?包|在哪|在那|在几楼|在几层|怎么走|带我去|方太|老板|三星|卫生间|厕所|积分|地铁|我要去|包装|停车场|电梯|出口|我想去|洗手间|充电|童车|贵吗|贵不贵|折扣吗|优惠吗|(有.*?吗)|要排队吗|人多吗).*?')
    qa_clean_pattern = re.compile(ur'在哪里|在哪|在那里|在那|怎么走|带我去下|带我去')
    greeting_pattern = re.compile(ur".*?(在吗|在嘛|名字|几岁|多少岁).*?")
    greeting_clean_pattern = re.compile(ur'啊|呢|呀')
    request_more_pattern = re.compile(ur".*?(换一|还有什么|还有其他|还有别的).*?")
    """
    return direction, suggested_direction, fixed q
    """
    def regex_plugin(self, q):
        # q = QueryUtils.static_corenlp_cut(q, remove_tags=QueryUtils.remove_tags)

        if re.match(self.request_more_pattern, q):
            return 'reqmore', None, q
        if re.match(self.base_pattern, q):
            return 'base', None, q
        type_ = self.entity_recog(q)
        if type_ == 'store':
            _, neg = self.neg.predict(q)
            if neg:
                return 'base', None, q
            else:
                return 'qa', None, q

        if type_ == 'item':
            return 'sale', 'qa', q

        try:
            if re.match(self.qa_pattern, q):
                # q = re.sub(self.qa_clean_pattern, '', q)
                return 'qa', None, q
            if re.match(self.sing_pattern, q):
                if re.match(self.sing_diff_pattern, q):
                    return 'qa', None, q
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

    graph_url = 'http://localhost:11403/solr/graph/select?q.op=OR&wt=json&q=%s'
    def entity_recog(self, q):
        try:
            _, type_,_ = self.retrieve_entity(q, type_='store')
            if type_:
                return type_
            _, type_, _ = self.retrieve_entity(q, type_='item')
            if type_:
                return type_
            return None
        except:
            traceback.print_exc()
            return None

    def retrieve_entity(self, q, type_ = None):
        if not type_:
            base_url = self.graph_url
        else:
            base_url = 'http://localhost:11403/solr/graph/select?q.op=OR&wt=json&q=%s AND type:' +  type_
        r = self._request_solr(q, 'name', base_url=base_url)
        if not r:
            return None, None, None
        name = SolrUtils.get_dynamic_response(r=r, key='name', random_field=True)
        type_ = SolrUtils.get_dynamic_response(r=r, key='type', random_field=True)
        if name:
            return name, type_, r
        else:
            return None, None, r

    def retrieve_label(self, q):
        r = self._request_solr(q, 'name', base_url=self.graph_url)
        name = SolrUtils.get_dynamic_response(r=r, key='label', random_field=True)
        if name:
            return name, 'item', r
        else:
            return None, None, None

    def _request_solr(self, q, key, base_url):
        ## cut q into tokens
        key = '%s:' % key
        tokens = [s for s in QueryUtils.static_jieba_cut(q, smart=False, remove_single=True)]
        if len(tokens) == 0:
            return None
        q = key + "(" + '%20'.join(tokens) + ")"
        url = base_url % q
        cn_util.print_cn(url)
        r = requests.get(url)
        return r

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
    # print(re.match(SceneKernel.qa_pattern, u'欧米茄在哪里'))
    print(SK.kernel(u'苹果'))