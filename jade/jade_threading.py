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
from queue import Queue
class MonitorLDKThread(Thread):
    def __init__(self,pyldk,JadeLog,ldkqueue,time=60*60,max_session_size=1):
        self.pyldk = pyldk
        self.JadeLog = JadeLog
        self.ldkqueue = ldkqueue
        self.time = time
        self.max_session_size = max_session_size
        self.handlequeue = Queue(maxsize=max_session_size)
        super(MonitorLDKThread, self).__init__()
        self.start()

    def logout(self):
        handle = self.handlequeue.get()
        self.pyldk.adapter.logout(handle)
    def run(self):
        haspStruct,feature_id = self.pyldk.login()
        if haspStruct.status == 0:
            self.handlequeue.put(haspStruct.handle)
        while haspStruct.status == 0:
            haspStruct, feature_id = self.pyldk.login()
            if haspStruct.status == 0:
                if self.handlequeue.qsize() == self.max_session_size:
                    self.logout()
                self.handlequeue.put(haspStruct.handle)
                if self.ldkqueue.qsize() > 0:
                    self.ldkqueue.get()
                self.ldkqueue.put((self.pyldk, haspStruct.handle))
            else:
                break
            if self.pyldk.get_ldk(feature_id) is False:
                break
            else:
                self.JadeLog.DEBUG("加密狗监听正常")
            time.sleep(self.time)

        self.JadeLog.ERROR("加密狗异常,程序退出")
        Exit(-800)
