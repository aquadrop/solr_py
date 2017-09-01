#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
First do some basic qa types like where, list, ...other give to question-question similarity module
"""
import requests
import random
import re
import traceback

import jieba

import numpy as np
import cPickle as pickle

from sc_qa_clf import SimpleSeqClassifier
from query_util import QueryUtils
from sc_simple_qa_kernel import SimpleQAKernel
from solr_utils import SolrUtils
import cn_util

class QAKernel:
    null_anwer = [u'啊呀！这可难倒宝宝了！这是十万零一个问题，你要问一下我们对面客服台的客服哥哥姐姐哦！']
    price_response = {u"奢侈":u"奢侈的东西,有钱人最爱",u"略贵":u"略贵",u"中档":u"还好,性价比高",u"便宜":u"很便宜的"}
    discount_response = {u"有折扣": u"有折扣的,快去店家看看吧", u"没有":u"可以看看别的商家"}
    queue_response = {u"排队": u"现在人有点多哦", u"不要排队": u"人不多,赶紧去吧"}
    static_clf = None
    # null_answer = ['null']
    def __init__(self):
        print('attaching qa kernel...')
        ## http://localhost:11403/solr/sc_qa/select?fq=entity:%E5%8E%95%E6%89%80&indent=on&q=*:*&wt=json
        self.graph_url = 'http://localhost:11403/solr/graph_v2/select?q.op=OR&wt=json&q=%s'
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
                return None, exact
            cls, probs = self.clf.predict(q)
            #
            if cls == 'where':
                direction, answer = self.where(q=q, last_r=last_r)
                return direction, answer
            if cls == 'exist':
                direction, answer = self.exist(q=q, last_r=last_r)
                return direction, answer
            if cls == 'ask_price':
                direction, answer = self.ask_price(q=q, last_r=last_r)
                return direction, answer
            if cls == 'ask_discount':
                direction, answer = self.ask_discount(q=q, last_r=last_r)
                return direction, answer
            if cls == 'ask_queue':
                direction, answer = self.ask_queue(q=q, last_r=last_r)
                return direction, answer
            # if cls == 'permit':
            #     direction, answer = self.permit(q=q, last_r=last_r)
            #     return direction, answer
            # if cls == 'whether':
            #     direction, answer = self.whether(q=q, last_r=last_r)
            #     return direction, answer
            # if cls == 'when':
            #     direction, answer = self.when(q=q, last_r=last_r)
            #     return direction, answer
            # if cls == 'how':
            #     direction, answer = self.how(q=q, last_r=last_r)
            #     return direction, answer
            # if cls == 'which':
            #     direction, answer = self.which(q=q, last_r=last_r)
            #     return direction, answer
            # if cls == 'what':
            #     direction, answer = self.what(q=q, last_r=last_r)
            #     return direction, answer
            if cls == 'list':
                direction, answer = self.list(q=q, last_r=last_r)
                return direction, answer
            return self.simple.kernel(q)
        except Exception,e:
            return self.simple.kernel(q)

    def common(self, q, key):
        r = self._request_solr(q, 'name', base_url=self.graph_url)

        try:
            num = SolrUtils.num_answer(r)
            if num > 0:
                response = SolrUtils.get_dynamic_response(r=r, key=key, random_field=True)
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
        current_entity, current_type, current_solr_r = self.retrieve_entity(q, 'store%20entertainment%20facility')
        if last_r and not current_entity:
            last_entity, last_type, last_solr_r = self.retrieve_entity(last_r, 'store%20entertainment%20facility')
        else:
            last_entity = None

        if current_entity:
            location = SolrUtils.get_dynamic_response(current_solr_r, 'rich_location', random_field=True)
            if not location:
                return self.simple.kernel(q)
            return None, current_entity + "," + location
        if not last_entity:
            location = '您在问什么?'
            return None, location

        q = QueryUtils.static_remove_pu(q).decode('utf-8')
        strict = re.compile(ur'在哪|在什么地方|带我去|在哪里')
        if re.match(strict, q):
            location = SolrUtils.get_dynamic_response(last_solr_r, 'rich_location', random_field=True)
            if not location:
                location = '数据库中不存在'
            return None, location
        return 'base', "数据库中不存在"

    def how(self, q, last_r):
        response = self.common(q, 'application')
        if response:
            return None, response
        return self.where(q, None)

    def when(self, q, last_r):
        response = self.common(q, 'time')
        if response:
            return None, response
        return self.simple.kernel(q)

    def what(self, q, last_r):
        r = self._request_solr(q, 'name', base_url=self.graph_url)
        response = SolrUtils.get_dynamic_response(r, 'definition')
        if response:
            return None, response
        return self.where(q, None)

    def list(self, q, last_r):
        current_entity, current_type, solr_r = self.retrieve_entity(q, type_='store')
        if current_entity:
            definition = SolrUtils.get_dynamic_response(r=solr_r, key='definition', random_field=True)
            if definition:
                return None, definition
            else:
                return None, '商家没有给出信息或者超出知识库范畴'

        current_entity, current_type, solr_r = self.retrieve_entity(q, type_='item')
        if current_entity:
            ## do listing
            rs = self._request_solr(current_entity, 'label', self.graph_url)
            response = SolrUtils.get_dynamic_response(r=rs, key='name',facet=True, random_field=True)
            if response and len(response) > 0:
                return None, ','.join(response)
        response = self.common(q, 'listing')
        if response:
            return None, response
        return self.simple.kernel(q)

    ## no logic reasoning
    def which(self, q, last_r):
        response = self.where(q)
        if response:
            return None, response
        return self.simple.kernel(q)

    ## --> exist: entity exists, return search. else return none
    def exist(self, q, last_r):
        current_entity, current_type, solr_r = self.retrieve_entity(q, 'store%20entertainment%20facility')

        if current_entity:
            response = self.retrieve_common_info(solr_r)
            if response:
                return None, current_entity + "," + response
            else:
                return None, '没有找到相关信息'

        current_label, current_type, label_r = self.retrieve_entity(q, 'item')
        ## hand over to main kernel
        if current_label:
            return 'sale', 'redirected'

        return 'base', 'redirected'

    ## --> exist: entity exists, return search. else return none
    def permit(self, q, last_r):
        current_entity, current_type, solr_r = self.retrieve_entity(q)

        if current_entity:
            r = self._request_solr(current_entity, 'name', base_url=self.graph_url)
            response = self.retrieve_common_info(r)
            if response:
                return None, response
            else:
                return None, '没有找到相关信息'

        current_label, current_type, label_r = self.retrieve_entity(q)
        ## hand over to main kernel
        if current_label:
            return self.simple(q)

        return None, '貌似不可以哦, 您可以去对面服务台咨询看看呢...'

    def ask_price(self, q, last_r):
        current_entity, current_type, solr_r = self.retrieve_entity(q, type_='store')
        if not current_entity and last_r:
            last_entity, last_type, last_solr_r = self.retrieve_entity(last_r, type_='store')
        else:
            last_entity = None

        use_entity = None
        use_type = None
        if current_entity:
            use_entity = current_entity
            use_type = current_type
            use_r = solr_r
        else:
            if last_entity:
                use_entity = last_entity
                use_type = last_type
                use_r = last_solr_r

        price = SolrUtils.get_dynamic_response(use_r, key='price', random_field=True)
        if price and use_entity and use_type == 'store':
            ## retrive labels
            return None, self.price_response[price.decode('utf-8')]

        return None, '不太清楚,请联系客服台或者商家咨询...'

    def ask_discount(self, q, last_r):
        current_entity, current_type, solr_r = self.retrieve_entity(q, type_='store')
        if not current_entity and last_r:
            last_entity, last_type, last_solr_r = self.retrieve_entity(last_r, type_='store')
        else:
            last_entity = None

        use_entity = None
        use_type = None
        if current_entity:
            use_entity = current_entity
            use_entity = current_type
            use_r = solr_r
        else:
            if last_entity:
                use_entity = last_entity
                use_type = last_type
                use_r = last_solr_r

        discount = SolrUtils.get_dynamic_response(use_r, key='discount', random_field=True)
        if discount and use_entity and use_type == 'store':
            ## retrive labels
            return None, self.discount_response[discount.decode('utf-8')]

        return None, '不太清楚,请联系客服台或者商家咨询...'

    def ask_queue(self, q, last_r, type_='store'):
        current_entity, current_type, solr_r = self.retrieve_entity(q, type_='store')
        if not current_entity and last_r:
            last_entity, last_type, last_solr_r = self.retrieve_entity(last_r, type_='store')
        else:
            last_entity = None

        use_entity = None
        use_type = None
        if current_entity:
            use_entity = current_entity
            use_entity = current_type
            use_r = solr_r
        else:
            if last_entity:
                use_entity = last_entity
                use_type = last_type
                use_r = last_solr_r

        queue = SolrUtils.get_dynamic_response(use_r, key='queue', random_field=True)
        if queue and use_entity and use_type == 'store':
            ## retrive labels
            return None, self.price_response[queue.decode('utf-8')]

        return None, '不太清楚,去商家看看呢...应该不用吧'

    ## --> exist: only the strict will be processed
    def whether(self, q, last_r):
        # current_entity, current_type, current_solr_r = self.retrieve_entity(q)
        # if last_r:
        #     last_entity, last_type, last_solr_r = self.retrieve_entity(last_r)
        # else:
        #     last_entity = None
        #
        # use_entity = None
        # if current_entity:
        #     if current_entity.decode('utf-8') == u'二期':
        #         return None, '是的'
        #     use_entity = current_entity
        # else:
        #     if last_entity:
        #         use_entity = last_entity
        #
        # if use_entity:
        #     valid = self.whether_label_validate(use_entity, q)
        #     if valid:
        #         return None, '恩'
        #     return None, '好像不是'
        #
        # return None, '我不知道,您可以去服务台问问哦'
        return self.simple.kernel(q)

    def whether_label_validate(self, entity, label_query):
        valid_url = 'http://localhost:11403/solr/graph_v2/select?&q=*:*&wt=json&fq=name:%s&fq=label:%s' % (entity, label_query)
        try:
            r = requests.get(valid_url)
            if SolrUtils.num_answer(r) > 0:
                return True
            else:
                return False
        except:
            return False

    def retrieve_common_info(self, r):
        location = SolrUtils.get_dynamic_response(r=r, key='rich_location', random_field=True)
        if location:
            return location
        definition = SolrUtils.get_dynamic_response(r=r, key='definition', random_field=True)
        if definition:
            return definition
        application = SolrUtils.get_dynamic_response(r=r, key='application', random_field=True)
        if definition:
            return application
        return None

    def retrieve_entity(self, q, type_ = None):
        if not type_:
            base_url = self.graph_url
        else:
            base_url = 'http://localhost:11403/solr/graph_v2/select?q.op=OR&wt=json&q={0}%20AND%20type:(' +  type_ + ")"
        r = self._request_solr(q, 'name', base_url=base_url)
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

    def exact_match(self, q, random_range=1):
        url = self.qa_exact_match_url % q
        r = requests.get(url)
        try:
            num = SolrUtils.num_answer(r)
            if num > 0:
                x = random.randint(0, min(random_range - 1, num - 1))
                response = SolrUtils.get_dynamic_response(r, x)
                return response
            else:
                return None
        except:
            return None

    def _extract_answer(self, r, random_range=1):
        try:
            num = SolrUtils.num_answer(r)
            if num > 0:
                x = random.randint(0, min(random_range - 1, num - 1))
                response = SolrUtils.get_dynamic_response(r, x)
                return True, response
            else:
                return False, np.random.choice(self.null_anwer, 1)[0]
        except:
            return False, np.random.choice(self.null_anwer, 1)[0]

    def _request_solr(self, q, key, base_url):
        try:
        ## cut q into tokens
            key = '%s:' % key
            tokens = [s for s in QueryUtils.static_jieba_cut(q, smart=False, remove_single=True)]
            q = key + "(" + '%20'.join(tokens) + ")"
            url = base_url.format(q)
            cn_util.print_cn(url)
            r = requests.get(url)
            return r
        except:
            traceback.print_exc()
            return None

    def _num_answer(self, r):
        return int(r.json()["response"]["numFound"])

if __name__ == '__main__':
    qa = QAKernel()
    # result = qa.kernel(u'三星手机在哪', u"Omega,一期三楼")
    cn_util.print_cn(qa.kernel(u'三楼我要去三楼', u"Omega,一期三楼")[1])
