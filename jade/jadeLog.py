#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : jadeLog.py
# @Author  : jade
# @Date    : 2019/12/19 13:56
# @Mailbox : jadehh@live.com
# @Software: Samples
# @Desc    :
import time

def JadeLog(content,DEUBG=False):
    if DEUBG:
        print("线上模型，应该保存log日志")
        with open("log.txt",'a') as f:
            f.write(">>{} [JadeLog INFO]: ".format( time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))+content+"\n")
    else:
        print(content)

if __name__ == '__main__':
    JadeLog("qqq,aaa,cccc",True)