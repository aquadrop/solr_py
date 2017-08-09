#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests
import json
import numpy as np

class BaseKernel:
    appid = "841e6cd456e05713213f413e8765648e"
    user_ids = np.array(['0112DBCD5299791D5A53287D27F4E18A5','0480704B8A3471FF360DD22AB5C3D9F8E','09B78AFBFCF3F97F34F12F945769FBD8B'])
    def __init__(self):
        print('attaching base kernel...')
        # if not self.user_ids or self.user_ids.size == 0:
        #     self.uid = self.register()
        # else:
        self.uid = self.user_ids[1]#np.random.choice(self.user_ids, 1)[0]

    def kernel(self, q):
        return self.chat(q)

    def register(self):
        register_data = {"cmd": "register", "appid": self.appid}
        url = "http://idc.emotibot.com/api/ApiKey/openapi.php"
        r = requests.post(url, params=register_data)
        response = json.dumps(r.json(), ensure_ascii=False)
        jsondata = json.loads(response)
        print("Return: %s" % jsondata.get('return'))
        print("ReturnMessage: %s" % jsondata.get('return_message'))
        datas = jsondata.get('data')
        for data in datas:
            print(data.get('value'))
            return data.get('value')

    def chat(self, q):
        response = None
        register_data = {"cmd": "chat", "appid": self.appid, "userid": self.uid, "text": q,
                         "location": "南京"}
        url = "http://idc.emotibot.com/api/ApiKey/openapi.php"
        r = requests.post(url, params=register_data)
        response = json.dumps(r.json(), ensure_ascii=False)
        jsondata = json.loads(response)
        print("Return: %s" % jsondata.get("return"))
        print("ReturnMessage: %s" % jsondata.get("return_message"))
        datas = jsondata.get("data")
        for data in datas:
            response = data.get('value')
            break
        return response

if __name__ == '__main__':
    bk = BaseKernel()
    # bk.register()
    bk.kernel(u'今天下雨吗')