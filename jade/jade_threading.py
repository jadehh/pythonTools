#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : jade_threading.py
# @Author   : jade
# @Date     : 2023/3/24 0024 16:34
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from threading import Thread
from jade.jade_tools import *
class MonitorLDKThread(Thread):
    def __init__(self,pyldk,JadeLog):
        self.pyldk = pyldk
        self.JadeLog = JadeLog
        super(MonitorLDKThread, self).__init__()
    def run(self):
        haspStruct,feature_id = self.pyldk.login()
        if haspStruct.status == 0:
            ldkqueue.put((self.pyldk,haspStruct.handle))
        while haspStruct.status == 0:
            if self.pyldk.get_ldk(feature_id) is False:
                break
            else:
                self.JadeLog.DEBUG("加密狗监听正常")
            time.sleep(60*60)
        self.JadeLog.ERROR("加密狗异常,程序退出")
        Exit(-800)
