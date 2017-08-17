#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This class is very simple and is stateless
"""
import requests
import random
import re

import jieba

import numpy as np
import cPickle as pickle

from sc_qa_clf import SimpleSeqClassifier
from query_util import QueryUtils
from sc_simple_qa_kernel import SimpleQAKernel
import cn_util

class QAKernel:
    null_anwer = ['啊呀！这可难倒宝宝了！这是十万零一个问题，你要问一下我们对面客服台的客服哥哥姐姐哦！']
    static_clf = None
    # null_answer = ['null']
    def __init__(self):
        print('attaching qa kernel...')
        ## http://localhost:11403/solr/sc_qa/select?fq=entity:%E5%8E%95%E6%89%80&indent=on&q=*:*&wt=json
        self.graph_url = 'http://localhost:11403/solr/graph/select?q.op=OR&wt=json&q=%s'
        self.qa_exact_match_url = 'http://localhost:11403/solr/sc_qa/select?wt=json&q=question:%s'
        self.simple = SimpleQAKernel()
        if QAKernel.static_clf:
            print('skipping attaching clf kernel...')
            self.clf = QAKernel.static_clf
        else:
            self.clf = SimpleSeqClassifier.get_instance('../model/sc/qa_clf.pkl')
            QAKernel.static_clf = self.clf

        # if QAKernel.static_belief_clf:
        #     print('skipping attaching clf kernel...')
        #     self.belief_clf = Multilabel_Clf.static_belief_clf
        # else:
        #     self.belief_clf = Multilabel_Clf.load('../model/sc/belief_clf.pkl')
        #     QAKernel.static_belief_clf = self.belief_clf

    ## classes: where,whether,when,how,which,what,list
    ## flow: check entity, check last_entity
    ## --> where: entity exists, return search. else strict use last_entity
    ## --> exist: entity exists, return search. else return none
    ## --> whether: entity exists, return search. else strict use last_entity
    def kernel(self, q, last_r):
        try:
            exact = self.exact_match(QueryUtils.static_remove_pu(q))
            if exact:
                return True, exact
            cls, probs = self.clf.predict(q)

            if cls == 'where':
                success, answer = self.where(q=q, last_r=last_r)
            if cls == 'exist':
                success, answer = self.exist(q=q, last_r=last_r)
            if cls == 'permit':
                success, answer = self.permit(q=q, last_r=last_r)
            if cls == 'whether':
                success, answer = self.whether(q=q, last_r=last_r)
            if cls == 'when':
                success, answer = self.when(q)
            if cls == 'how':
                success, answer = self.how(q)
            if cls == 'which':
                success, answer = self.which(q)
            if cls == 'what':
                success, answer = self.what(q)
            if cls == 'list':
                success, answer = self.list(q)
            return success, answer
        except Exception,e:
            return self.simple.kernel(q)

    def common(self, q, key):
        r = self._request_solr(q, 'name')

        try:
            num = self._num_answer(r)
            if num > 0:
                response = self._get_response(r=r, key=key, random=True)
                if response:
                    return response
                else:
                    return None
            else:
                return None
        except:
            return None

    ## where: entity exists, return search. else strict use last_entity
    def where(self, q, last_r):
        current_entity = self.retrieve_entity(q)
        if last_r:
            last_entity = self.retrieve_entity(last_r)
        else:
            last_entity = None

        if current_entity:
            location = self.common(current_entity, 'location')
            if not location:
                location = '没有这个地方哦'
            return True, location
        if not last_entity:
            location = '您在问什么?'
            return True, location

        q = QueryUtils.static_remove_pu(q).decode('utf-8')
        strict = re.compile(ur'在哪|在什么地方|带我去|在哪里')
        if re.match(strict, q):
            location = self.common(last_entity, 'location')
            if not location:
                location = '没有这个地方哦'
            return True, location
        return True, "没有这个地方哦"

    def how(self, q, last_r):
        response = self.common(q, 'application')
        if response:
            return True, response
        return True, np.random.choice(self.null_anwer, 1)[0]

    def when(self, q, last_r):
        response = self.common(q, 'time')
        if response:
            return True, response
        return False, np.random.choice(self.null_anwer, 1)[0]

    def what(self, q, last_r):
        response = self.common(q, 'definition')
        if response:
            return True, response
        return True, np.random.choice(self.null_anwer, 1)[0]

    def list(self, q, last_r):
        response = self.common(q, 'listing')
        if response:
            return True, response
        return True, "商家没有给出信息,请前往商家咨询"

    ## no logic reasoning
    def which(self, q, last_r):
        response = self.where(q)
        if response:
            return True, response
        return False, np.random.choice(self.null_anwer, 1)[0]

    ## --> exist: entity exists, return search. else return none
    def exist(self, q, last_r):
        current_entity = self.retrieve_entity(q)

        if current_entity:
            r = self._request_solr(current_entity, 'name')
            response = self.retrieve_common_info(r)
            if response:
                return True, response
            else:
                return True, '没有找到相关信息'

        current_label = self.retrieve_label(q)
        ## hand over to main kernel
        if current_label:
            return False, None

        return True, '貌似没有哦, 您可以去对面服务台咨询看看呢...'

    ## --> exist: entity exists, return search. else return none
    def permit(self, q, last_r):
        current_entity = self.retrieve_entity(q)

        if current_entity:
            r = self._request_solr(current_entity, 'name')
            response = self.retrieve_common_info(r)
            if response:
                return True, response
            else:
                return True, '没有找到相关信息'

        current_label = self.retrieve_label(q)
        ## hand over to main kernel
        if current_label:
            return False, None

        return True, '貌似不可以哦, 您可以去对面服务台咨询看看呢...'


    ## --> exist: only the strict will be processed
    def whether(self, q, last_r):
        current_entity = self.retrieve_entity(q)
        if last_r:
            last_entity = self.retrieve_entity(last_r)
        else:
            last_entity = None

        use_entity = None
        if current_entity:
            use_entity = current_entity
        else:
            if last_entity:
                use_entity = last_entity

        if use_entity:
            ## retrive labels
            price_high = re.compile(ur'.*?(贵).*?')
            price_low = re.compile(ur'.*?(实惠|便宜).*?')
            discount = re.compile(ur'.*?(优惠|折扣|打折).*?')
            if re.match(price_high, q):
                q = '高'
                valid = self.whether_label_validate(use_entity, q)
                if valid:
                    return True, '有点小贵哦'
                else:
                    return True, '不贵很实惠'
            if re.match(price_low, q):
                q = '低'
                valid = self.whether_label_validate(use_entity, q)
                if valid:
                    return True, '不贵'
                else:
                    return True, '有点小贵哦'
            if re.match(discount, q):
                q = '有'
                valid = self.whether_label_validate(use_entity, q)
                if valid:
                    return True, '有优惠的,具体请去店家看看吧'
                else:
                    return True, '没有优惠哦,具体请去店家看看吧'
            valid = self.whether_label_validate(use_entity, q)
            if valid:
                return True, '恩'
            return True, '好像没有'

        return True, '我不知道,您可以去服务台问问哦'

    def whether_label_validate(self, entity, label_query):
        valid_url = 'http://localhost:11403/solr/graph/select?&q=*:*&wt=json&fq=name:%s&fq=label:%s' % (entity, label_query)
        try:
            r = requests.get(valid_url)
            if self._num_answer(r) > 0:
                return True
            else:
                return False
        except:
            return False

    def retrieve_common_info(self, r):
        location = self._get_response(r=r, key='location', random=True)
        if location:
            return location
        definition = self._get_response(r=r, key='definition', random=True)
        if definition:
            return definition
        application = self._get_response(r=r, key='application', random=True)
        if definition:
            return application
        return None

    def retrieve_entity(self, q):
        name = self.common(q, key='name')
        if name:
            return name.split(',')[0]
        else:
            return None

    def retrieve_label(self, q):
        name = self.common(q, key='label')
        if name:
            return name.split(',')[0]
        else:
            return None

    def exact_match(self, q, random_range=1):
        url = self.qa_exact_match_url % q
        r = requests.get(url)
        try:
            num = self._num_answer(r)
            if num > 0:
                x = random.randint(0, min(random_range - 1, num - 1))
                response = self._get_response(r, x)
                return response
            else:
                return None
        except:
            return None

    def _extract_answer(self, r, random_range=1):
        try:
            num = self._num_answer(r)
            if num > 0:
                x = random.randint(0, min(random_range - 1, num - 1))
                response = self._get_response(r, x)
                return True, response
            else:
                return False, np.random.choice(self.null_anwer, 1)[0]
        except:
            return False, np.random.choice(self.null_anwer, 1)[0]

    def _request_solr(self, q, key):
        ## cut q into tokens
        key = '%s:' % key
        tokens = [key + s for s in QueryUtils.static_jieba_cut(q, False)]
        q = ' OR '.join(tokens)
        url = self.graph_url % q
        # print('qa_debug:', url)
        cn_util.print_cn(url)
        r = requests.get(url)
        return r

    def _num_answer(self, r):
        return int(r.json()["response"]["numFound"])

    def _get_response(self, r, key, random=True, keep_array=False):
        try:
            a = r.json()["response"]["docs"][0][key]
            if keep_array:
                return a
            else:
                if random:
                    rr = np.random.choice(a, 1)[0]
                else:
                    rr = ','.join(a)
            return rr.encode('utf8')
        except:
            return None

if __name__ == '__main__':
    qa = QAKernel()
    cn_util.print_cn(qa.kernel(u'打折吗',u'NIKE运动鞋、篮球鞋、休闲鞋任您选，快去一期五楼看看吧，just do it'))