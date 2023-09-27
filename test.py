#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : test.py
# @Author   : jade
# @Date     : 2023/3/7 11:16
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from jade import *
from jade.jade_threading import MonitorLDKThread
from pyldk.pyldk import PyLdk

def test_process_bar():
    progressBar = ProgressBar(10)
    for i in range(10):
        time.sleep(1)
        progressBar.update()
def test_monitor_pydk():
    JadeLog  = JadeLogging("log",Level="DEBUG")
    ldkqueue = Queue(10)
    LDKREFRESHTIME = 30
    MonitorLDKThread(PyLdk(JadeLog), JadeLog, ldkqueue, LDKREFRESHTIME)

def test_print_a():
    a = b'\xe7\x9b\xb8\xe6\x9c\xba\xe5\xbc\x82\xe5\xb8\xb8'
    print(a.decode("utf-8"))


if __name__ == '__main__':
    count = 0


    def increase_count(count):
        count += 1


    print(count)  # 输出0
    increase_count(count)
    print(count)