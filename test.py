#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : test.py
# @Author   : jade
# @Date     : 2024/2/27 9:16
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from dataset_tools.jade_create_object_dection_datasets import *


def testCreateYearsDatasets():
    """
    制作Paddle Voc 数据集
    """
    CreateYearsDatasets(r"F:\数据集\VOC数据集\验残集装箱检测数据集",  save_path=r"E:\Data\VOC数据集\验残集装箱检测数据集")


if __name__ == '__main__':
    testCreateYearsDatasets()
