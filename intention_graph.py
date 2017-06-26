import sys
import json
import _uniout
import cPickle as pickle
reload(sys)
sys.setdefaultencoding("utf-8")


def intention_graph(path):
    with open(path, "r") as file:
        lines = file.readlines()
    texts = list()
    intention_list = list()
    for line in lines:
        text = json.loads(line)["intention_list"]
        texts.append(text)
    # print _uniout.unescape(str(texts), 'utf8')
    for intentions in texts:
        for intention in intentions:
            if intention not in intention_list:
                intention_list.append(intention)
    # print len(intention_list)
    # print _uniout.unescape(str(intention_list), 'utf8')
    intention_graph = dict()
    for intention in intention_list:
        key = intention
        value = list()
        for x in texts:
            if key in x and x.index(key) + 1 < len(x) and x[x.index(key) + 1] not in value:
                value.append(x[x.index(key) + 1])
        intention_graph[key] = value
    print _uniout.unescape(str(intention_graph), 'utf8')
    return intention_graph


if __name__ == '__main__':
    intention_graph = intention_graph("data/data.txt")
    with open("data/intention_graph", "w+") as f:
        json.dump(intention_graph, f, ensure_ascii=False)
