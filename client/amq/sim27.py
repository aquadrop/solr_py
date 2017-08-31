# -*- coding:utf-8 -*-
import sys
from IMQ import IMessageQueue
import json
import time
import random_helper

class BenebotSim():

    mq = None #messagequeue

    def __init__(self):
        publish_key = 'nlp.sim.normal.request.'+random_helper.random_string()
        receive_key = publish_key.replace('request', 'reply')
        self.mq = IMessageQueue(receive_key, publish_key, receive_key, receive_key, '')

    def getMaxSim(self, sequence, sentences):
        if not sequence or not sentences:
            return {}
        #st = time.time()
        result = self.mq.request_synchronize(json.dumps({'sequence': sequence, 'sentences': sentences}))
        #print('time: ', time.time()-st)
        if result:
            return json.loads(result)
        return {}

if __name__ == '__main__':
    bt = BenebotSim()
    while 1:
        s1 = raw_input('input: ')
        s2 = raw_input('input: ')
        result = bt.getMaxSim(s1, [s2])
        print result
