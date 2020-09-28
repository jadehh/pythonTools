#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : jadeLogger.py
# @Author  : jade
# @Date    : 2020/9/24 上午9:39
# @Mailbox : jadehh@live.com
# @Software: Samples
# @Desc    :
from jade import *

class JadeLogger(Thread):
    def __init__(self,logger_path):
        self.logger_path = logger_path
        CreateSavePath(self.logger_path)
        self.logContentQueue = Queue(maxsize=100)
        self.create_today = GetToday().split("-")[-1]
        if self.create_today[0] == "0":
            self.create_today = int(self.create_today[1])
        else:
            self.create_today = int(self.create_today)
        Thread.__init__(self)
        self.start()

    def INFO(self,content):
        content =  content
        self.logContentQueue.put(content)

    def ERROR(self,content):
        self.logContentQueue.put(content)


    def get_today(self):
        this_day = GetToday().split("-")[-1]
        if this_day[0] == "0":
            this_day = int(this_day[1])
        else:
            this_day = int(this_day)
        return this_day
    def run(self):
        while True:
            day = self.get_today()
            if day- self.create_today == 1:
                """
                分割info.log,
                复制一份,info.log重新开始记录
                """
            content = self.logContentQueue.get()
            with open(os.path.join(self.logger_path,"info.log"),"a") as f:
                f.write(content+"\n")

if __name__ == '__main__':
    jadeLog = JadeLogger("log")

    jadeLog.INFO("haha")