#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from client.sc.scene_clf import SceneClassifier

class SceneKernel:
    def __init__(self):
        try:
            print('attaching scene kernel...')
            self.clf = SceneClassifier.get_instance('../../model/sc/scene_clf.pkl')
        except Exception,e:
            print('failed to attach scene kernel..all inquires will be redirected to main kernel..', e.message)

    def kernel(self, q):
        if not self.clf:
            return None
        labels, _ = self.clf.predict(question=q)
        return self.select_label(labels)

    def select_label(self, labels):
        """
        sale > qa > repeat > greetings = base
        :param labels:
        :return:
        """

        if 'sale' in labels:
            return 'sale'
        else:
            if 'qa' in labels:
                return 'qa'
            else:
                if 'repeat_machine' in labels:
                    return 'repeat_machine'
                if 'repeat_user' in labels:
                    return 'repeat_user'
                else:
                    return np.random.choice(labels, 1)[0]


if __name__ == '__main__':
    SK = SceneKernel()
    print(SK.kernel('你叫什么名字'))