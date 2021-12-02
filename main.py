#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : main.py
# @Author   : dataset_tools
# @Date     : 2021/5/6 9:38
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from dataset_tools.jade_create_paddle_text_detection_datasets import *
if __name__ == '__main__':
    create_text_detection_datasets(r"F:\数据集\关键点检测数据集\箱号关键点数据集",r"E:\Data\字符检测识别数据集\箱号关键点数据集",0.95)
