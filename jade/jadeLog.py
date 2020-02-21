#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : jadeLog.py
# @Author  : jade
# @Date    : 2019/12/19 13:56
# @Mailbox : jadehh@live.com
# @Software: Samples
# @Desc    :
import time

def JadeLog(content,log_path,DEUBG=False):
    if DEUBG:
        with open(log_path,'a') as f:
            print(">>{} [JadeLog INFO]: ".format( time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))+content)
            f.write(">>{} [JadeLog INFO]: ".format( time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))+content+"\n")
    else:
        print(">>{} [JadeLog INFO]: ".format( time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))+content)

if __name__ == '__main__':
    JadeLog("qqq,aaa,cccc",True)