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
if __name__ == '__main__':
    test_monitor_pydk()