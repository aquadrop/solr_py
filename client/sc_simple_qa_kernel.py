#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This class is very simple and is stateless
"""
import requests
import random

import numpy as np

from query_util import QueryUtils
import cn_util
from solr_utils import SolrUtils
from amq.sim27 import BenebotSim

class SimpleQAKernel:
    null_anwer = ['啊呀！这可难倒宝宝了！这是十万零一个问题，你要问一下我们对面客服台的客服哥哥姐姐哦！']
    # null_answer = ['null']

    def __init__(self):
        print('attaching qa kernel...')
        ## http://localhost:11403/solr/sc_qa/select?fq=entity:%E5%8E%95%E6%89%80&indent=on&q=*:*&wt=json
        self.qa_url = 'http://localhost:11403/solr/sc_qa/select?q.op=OR&wt=json&q={0}'
        self.qa_exact_match_url = 'http://localhost:11403/solr/sc_qa/select?wt=json&q=question:{0}'
        self.qu = QueryUtils()
        self.bt = BenebotSim.Instance()

    def kernel(self, q):
        try:
            exact = self.exact_match(q)
            if exact:
                return None, exact
            r = self._request_solr(q)
            direction, answer = self._extract_answer(q, r)
            return direction, answer
        except Exception,e:
            return None, 'mainframe unable to reply since qa kernel damaged...'

    def exact_match(self, q, random_range=1):
        url = self.qa_exact_match_url.format(q)
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

    def _extract_answer(self, q, r, random_range=1):
        try:
            num = self._num_answer(r)
            if num > 0:
                response = self.select_max_match_with_sim(q, r)
                if response:
                    return None, response
                else:
                    return 'sale', response
            else:
                return None, np.random.choice(self.null_anwer, 1)[0]
        except:
            return None, np.random.choice(self.null_anwer, 1)[0]

    def _request_solr(self, q):
        ## cut q into tokens
        tokens = ['question:' + s for s in QueryUtils.static_jieba_cut(q, smart=False, remove_single=True)]
        q = ' OR '.join(tokens)
        url = self.qa_url.format(q)
        # print('qa_debug:', url)
        cn_util.print_cn(url)
        r = requests.get(url)
        return r

    def select_max_match_with_sim(self, q, r):
        matched_questions = SolrUtils.get_dynamic_response(r=r, key='question',
                                                           random_hit=False,
                                                           random_field=True,
                                                           keep_array=False,
                                                           facet=True)
        q_tokens = ' '.join(QueryUtils.static_jieba_cut(q))
        matched_questions_tokens = [' '.join(QueryUtils.static_jieba_cut(mqt)) for mqt in matched_questions]
        max_sim = self.bt.getMaxSim(q_tokens, matched_questions_tokens)
        best_sentence = ''.join(max_sim['sentence'].split(' '))
        sim = max_sim['sim']
        cn_util.print_cn(best_sentence, str(sim), '[' + ','.join(matched_questions) + ']')
        if sim > 0.3:
            index = matched_questions.index(best_sentence)
            answer = SolrUtils.get_dynamic_response(r, key='answer', force_hit=index,
                                                    random_field=True,
                                                    random_hit=False)
            return answer
        return None

    def _num_answer(self, r):
        return int(r.json()["response"]["numFound"])

    def _get_response(self, r, i=0):
        try:
            a = r.json()["response"]["docs"][i]['answer']
            rr = np.random.choice(a, 1)[0]
            x = random.randint(0, min(0, len(a) - 1))
            return rr.encode('utf8')
        except:
            return None

    def purify_q(self, q):
        q = self.qu.remove_cn_punct(q)
        pos_q = self.qu.corenlp_cut(q, remove_tags=["CD", "VA", "AD", "VC"])
        return ''.join(pos_q), q

if __name__ == '__main__':
    qa = SimpleQAKernel()
    cn_util.print_cn(qa.kernel(u'得基怎么去')[1])