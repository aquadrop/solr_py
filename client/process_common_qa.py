#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
import _uniout
import xlrd
import uuid


reload(sys)
sys.setdefaultencoding("utf-8")


def process(path, output):
    data = xlrd.open_workbook(path)

    with open(output, 'w+') as out:
        for i in xrange(len(data.sheets())):
            table = data.sheets()[i]
            rows = table.nrows
            for j in xrange(rows):
                if len(table.row_values(j)[0]) == 0:
                    continue
                print table.row_values(j)
                qa = {'question': table.row_values(j)[
                    0], 'answer': table.row_values(j)[1],
                    'id': str(uuid.uuid1())}
                out.write(json.dumps(qa))
                out.write(',\n')


if __name__ == '__main__':
    process('../data/common_qa.xlsx', '../data/common_qa.json')
