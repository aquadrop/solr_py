#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import csv

import jieba

import cn2arab
import cn_util
import re

class QueryUtils:
    static_tokenizer_url = "http://localhost:11415/pos?q="

    def __init__(self):
        self.remove_tags = ["PN", "VA", "AD"]
        self.tokenizer_url = "http://localhost:11415/pos?q="

    def jieba_cut(self, query):
        seg_list = jieba.cut(query)
        tokens = []
        for t in seg_list:
            tokens.append(t)
        return tokens

    def corenlp_cut(self, query, remove_tags=[]):
        q = query
        r = requests.get(url=self.tokenizer_url + q)
        ## purify
        text = []
        for t in r.text.encode("utf-8").split(" "):
            tag = t.split("/")[1]
            word = t.split("/")[0]
            if not tag in remove_tags:
                text.append(word)
        return text

    @staticmethod
    def static_corenlp_cut(query, remove_tags=[]):
        q = query
        r = requests.get(url=QueryUtils.static_tokenizer_url + q)
        ## purify
        text = []
        for t in r.text.encode("utf-8").split(" "):
            tag = t.split("/")[1]
            word = t.split("/")[0]
            if not tag in remove_tags:
                text.append(word)
        return text

    def pos(self, query, remove_tags=[]):
        try:
            q = query
            r = requests.get(url=self.tokenizer_url+q)
            ## purify
            text = []
            for t in r.text.encode("utf-8").split(" "):
                tag = t.split("/")[1]
                if not tag in remove_tags:
                    text.append(t)
            return text
        except:
            return [query]

    skip_CD = ['一些','一点','一些些','一点点','一点零']

    def quant_fix(self, query):
        pos = self.pos(query)
        fixed = False
        new_query = []
        for t in pos:
            word, tag = t.split("/")
            if tag == 'CD' and word not in self.skip_CD:
                _, word = cn2arab.cn2arab(word)
                fixed = True
            new_query.append(word)
        return fixed, new_query

    def quant_bucket_fix(self, query):
        b, q = self.quant_fix(query)
        if b:
            new_q = []
            for token in q:
                if token.isdigit():
                    token = self.fix_money(token)[0]
                new_q.append(token)
            return new_q
        return query

    def remove_cn_punct(self, q):
        return ''.join(self.corenlp_cut(q, remove_tags=['PU']))

    @staticmethod
    def static_remove_cn_punct(q):
        return ''.join(QueryUtils.static_corenlp_cut(q, remove_tags=['PU']))


    tokenizer_url = "http://localhost:11415/pos?q="
    transfer_ = {1: '零钱', 200: ' 二百 ', 20000: ' 二万 ', 50000: ' 五万 ', 1000000: ' 一百万 '}
    # transfer_ = {200: ' 200 ', 20000: ' 20000 ', 50000: ' 50000 ', 1000000: ' 1000000 '}
    ## 0-20K,20K-50K,50K-100K,100K-
    breakpoints_map = {"转账": [-10, 0, 49999, 1000000 - 1, 9999999999999],
                       "存款": [-10, 0, 199, 50000 - 1, 9999999999999],
                       "取款": [-10, 0, 199, 20000 - 1, 50000 - 1, 9999999999999]}

    def transfer(self, values):
        return [self.transfer_[value] for value in values]

    def fix_money(self, money, slot=None):
        num = cn2arab.cn2arab_core(money)

        values = []
        if slot is None:
            if num >= 0 and num <= 99:
                values = [1]
            if num >= 100 and num <= 19999:
                values = [200]
            elif num >= 20000 and num <= 49999:
                values = [20000]
            elif num >= 50000 and num <= 999999:
                values = [50000]
            elif num >= 1000000:
                values = [1000000]
            return self.transfer(values)

        ## 转账
        ## 0-20K,20K-50K,50K-100K,100K-
        if "转账" in slot:
            if num >= 0 and num <= 49999:
                values = [1, 200, 20000]
            elif num >= 50000 and num <= 1000000 - 1:
                values = [50000]
            elif num >= 1000000 and num <= 9999999999999:
                values = [1000000]
            return self.transfer(values)

        ## 存款
        ## 0-20K,20K-50K,50K-100K,100K-
        if "存款" in slot:
            if num >= 0 and num < 199:
                values = [1]
            if num >= 200 and num <= 49999:
                values = [200, 20000]
            elif num >= 50000 and num <= 9999999999999:
                values = [50000, 1000000]
            return self.transfer(values)

        ## 取款
        ## 0-20K,20K-50K,50K-100K,100K-
        if "取款" in slot:
            if num >= 0 and num < 100:
                values = [1]
            if num >= 100 and num <= 19999:
                values = [200]
            elif num >= 20000 and num <= 49999:
                values = [20000]
            elif num >= 50000 and num <= 9999999999999:
                values = [50000, 1000000]
            return self.transfer(values)

    def all_possible(self, tokens, slot):
        if len(tokens) == 0:
            return []
        token_head = tokens[0]
        word = token_head
        heads = []
        if word.isdigit():
            fixed_words = self.fix_money(word, slot)
            for w in fixed_words:
                heads.append(w)
        else:
            heads = [word]

        others = self.all_possible(tokens[1:], slot)

        possibles = []
        if len(others) > 0:
            for head in heads:
                for other in others:
                    possible = head + other
                    possibles.append(possible)
        else:
            possibles = heads

        return possibles

    def make(self, q, slot):
        b, q = self.quant_fix(q)
        if b:
            cn_util.print_cn(str(q))
            r = self.jieba_cut(''.join(q))
            # purify
            return self.all_possible(r, slot)
        else:
            return None

    def process_data(self, path, outpath):
        with open(outpath, 'w+') as out:
            with open(path, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for line in reader:
                    write = [line]
                    response = None
                    if u'存款' in line[0]:
                        q = line[1].encode('utf-8')
                        response = self.make(q, '存款')
                    elif u'取款' in line[0]:
                        q = line[1].encode('utf-8')
                        response = self.make(q, '取款')
                    elif u'转账' in line[0]:
                        q = line[1].encode('utf-8')
                        response = self.make(q, '转账')

                    if response:
                        write = []
                        for r in response:
                            w = [line[0], r.strip()]
                            write.append(w)

                    for w in write:
                        ww = '\t'.join(w)
                        mm = ww.strip()
                        out.write(mm + '\n')

if __name__ == '__main__':
    qu = QueryUtils()
    # qu.process_data('../data/train_pruned.txt', '../data/train_pruned_fixed2.txt')

    print(QueryUtils.static_remove_cn_punct(u'我在电视上见过你，听说你很聪明啊?'))
    cn_util.print_cn(qu.quant_bucket_fix('我要取三百'))