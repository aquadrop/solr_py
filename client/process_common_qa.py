#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
import _uniout
import xlrd
import uuid


reload(sys)
sys.setdefaultencoding("utf-8")


def process(path, output, output2):
    data = xlrd.open_workbook(path)

    with open(output, 'w+') as out:
        with open(output2, 'w+') as out2:
            for i in xrange(len(data.sheets())):
                table = data.sheets()[i]
                rows = table.nrows
                for j in xrange(rows):
                    if len(table.row_values(j)[0]) == 0:
                        continue
                    question = table.row_values(j)[0].replace('\n', '')
                    answer = table.row_values(j)[1].replace('\n', '')
                    qa = {'question': question, 'answer': answer,
                          'id': str(uuid.uuid1())}
                    line = answer + '\t' + question
                    out2.write(line)
                    out2.write('\n')
                    out.write(json.dumps(qa))
                    out.write(',\n')


if __name__ == '__main__':
    process('../data/common_qa.xlsx',
            '../data/common_qa.json', '../data/common_qa.txt')
