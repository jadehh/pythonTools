#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : jadeTools.py
# @Author   : jade
# @Date     : 2021/5/25 10:28
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
def zh_ch(string):
    """
    解决cv2.namedWindow中文乱码问题
    :param string:
    :return:
    """
    return string.encode("gbk").decode('UTF-8', errors='ignore')

def getNumberofString(string):
    """
    提取字符串中的数字
    :param string:
    :return:
    """
    return "".join(list(filter(str.isdigit, string)))