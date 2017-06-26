#!/usr/bin/env python
# encoding: utf-8
import sys
import json
import uuid
import pysolr

import _uniout
reload(sys)
sys.setdefaultencoding("utf-8")


json_data_path = 'data.txt'
yingxiao_data_path = 'test.txt'


def preprocess(lines):
    num_list = ["一", "二", "三", "四", "五", "六", "七", "八", "九"]
    new_lines = list()
    with open("pre.txt", "w+") as f:
        for line in lines:
            line = line.strip().replace(" ", "").replace("\t", "")
            if len(line) and line[:3] not in num_list:
                # print line
                # print len(line)
                f.write(line + "\n")
                new_lines.append(line)
    return new_lines


def parse_rawdata(inpath, outpath):
    num_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    with open(outpath, 'w+') as out:
        with open(inpath, 'r') as file:
            lines = file.readlines()
            lines = preprocess(lines)
            i = 0
            while i < len(lines):
                current = i
                j = current + 1
                while j < len(lines) and lines[j][0] not in num_list:
                    j += 1
                pos_G = lines[i].find("G")
                pos_B = lines[i].find("B")
                if pos_G >= 0:
                    lines[i] = lines[i][pos_G:]
                elif pos_B >= 0:
                    lines[i] = lines[i][pos_B:]
                else:
                    i += 1

                question_list = list()
                answer_list = list()
                for x in xrange(i, j):
                    if lines[i].find("B") >= 0:
                        continue
                    if lines[x].find("B") >= 0:
                        answer_list.append(lines[x])
                    else:
                        question_list.append(lines[x])
                    length = min(len(question_list), len(answer_list))
                    for y in xrange(0, length):
                        result = dict()
                        result["id"] = str(uuid.uuid1())
                        result["question"] = question_list[y]
                        result["answer"] = answer_list[y]
                        if y == 0:
                            result["last_question"] = ""
                            result["last_answer"] = ""
                        else:
                            result["last_intention"] = question_list[y - 1]
                       
                        if y == (length - 1):
                            result["next_intention"] = ""
                        
                        else:
                            result["next_intention"] = question_list[y + 1]

                        out.write(json.dumps(
                            result, ensure_ascii=False) + "," + "\n")

                i = j


def parse_json(inpath, outpath):
    with open(outpath, 'w+') as out:
        with open(inpath, 'r') as file:
            line_num = 0
            for line in file:
                print "line %d" % line_num
                text = json.loads(line)
                question_list = text["question_list"]
                answer_list = text["answer_list"]
                business_list = text["business_list"]
                intension_list = text["intension_list"]

                for i in xrange(0, len(question_list)):
                    print "question %d of %d" % (i, line_num)
                    result = dict()
                    result["id"] = str(uuid.uuid1())
                    result["question"] = question_list[
                        i]
                    result["answer"] = answer_list[i]
                    result["intension"] = intension_list[i]
                    result["business"] = business_list[i]
                    result["amount_lower"] = 0.0
                    result["amount_upper"] = 0.0

                    if i == 0:
                        result["last_intension"] = ""
                        result["last_business"] = ""
                    else:
                        result["last_intension"] = intension_list[i - 1]
                        result["last_business"] = business_list[i - 1]

                    if i == (len(question_list) - 1):
                        result["next_intension"] = ""
                        result["next_business"] = ""
                    else:
                        result["next_intension"] = intension_list[i + 1]
                        result["next_business"] = business_list[i + 1]

                    # print type(json.dumps(result))
                    out.write(json.dumps(
                        result, ensure_ascii=False) + "," + "\n")
                line_num += 1


if __name__ == '__main__':
    parse_rawdata("chats.txt", "chats.json")
