#!/usr/bin/env python
# encoding: utf-8
import sys
import json
import uuid

import _uniout
reload(sys)
sys.setdefaultencoding("utf-8")


data_path = 'data.txt'


def parse_json(path):
    with open('dump2.txt', 'w+') as out:
        with open(path, 'r') as file:
            line_num = 0
            for line in file:
                print "line %d" % line_num, line
                text = json.loads(line)
                question_list = text["question_list"]
                answer_list = text["answer_list"]
                business_list = text["business_list"]
                intention_list = text["intention_list"]

                for i in xrange(0, len(question_list)):
                    print "question %d of %d" % (i, line_num)
                    result = dict()
                    result["id"] = str(uuid.uuid1())
                    result["question"] = question_list[
                        i].encode("utf-8")
                    result["answer"] = answer_list[i]
                    result["intention"] = intention_list[i]
                    result["business"] = business_list[i]
                    result["amount_lower"] = 0.0
                    result["amount_upper"] = 0.0

                    if i == 0:
                        result["last_intention"] = ""
                        result["last_business"] = ""
                    else:
                        result["last_intention"] = intention_list[i - 1]
                        result["last_business"] = business_list[i - 1]

                    if i == (len(question_list) - 1):
                        result["next_intention"] = ""
                        result["next_business"] = ""
                    else:
                        result["next_intention"] = intention_list[i + 1]
                        result["next_business"] = business_list[i + 1]

                    result = parse_intention(result['intention'], result)
                    # print type(json.dumps(result))
                    out.write(json.dumps(
                        result, ensure_ascii=False) + "," + "\n")
                line_num += 1

def parse_intention(intention, result):
	if u"两万以下" in intention:
		result['amount_lower'] = 0
		result['amount_upper'] = 20000
	if u"五万以上" in intention:
		result['amount_lower'] = 50000
		result['amount_upper'] = 100000000
	if u"五万以下" in intention:
		result['amount_lower'] = 0
		result['amount_upper'] = 50000
	if u"两万到五万" in intention:
		result['amount_lower'] = 20000
		result['amount_upper'] = 50000
	if u"五万到一百万" in intention:
		result['amount_lower'] = 50000
		result['amount_upper'] = 1000000
	if u"一百万以上" in intention:
		result['amount_lower'] = 1000000
		result['amount_upper'] = 100000000
	return result



def test_unicode():
    with open('test_dump2.txt', 'w+') as out:
        with open('test_data.txt', 'r') as file:
            for line in file:
                text = json.loads(line)
                result = dict()
                question_list = text["question_list"]
                s = question_list[0]
                # print type(s)
                # s = s.decode("utf-8")
                print _uniout.unescape(s, "utf-8")
                result["question"] = s.decode("utf-8")
                out.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    parse_json(data_path)
    # test_unicode()
