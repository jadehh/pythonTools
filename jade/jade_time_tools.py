#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : jade_time_tools.py
# @Author   : jade
# @Date     : 2021/10/27 16:30
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
import time
"""
时间戳转字符串 
"""
def time_to_str(timeStamp:float):
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime
