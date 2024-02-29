#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : test.py
# @Author   : jade
# @Date     : 2024/2/27 9:16
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from dataset_tools.jade_create_object_dection_datasets import *

if __name__ == '__main__':
    CreateYearsDatasets("F:\数据集\VOC数据集\箱门检测数据集\ContainVOC",year=None,save_path="E:\数据集\VOC数据集\箱门检测数据集\ContainVOC",rate=0.95)
