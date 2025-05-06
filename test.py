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
    VOC_CLASSES = ["FRONTEND","DOOREND","UPEND","slide","bromine_tank"]
    CreateYearsDatasets(r"F:\数据集\VOC数据集\箱门检测数据集\ContainVOC",  save_path=r"E:\Data\VOC数据集\箱门检测数据集\ContainVOC")

def testCreateYearsDarknetVocDatasets():
    # VOC_CLASSES = ["container"]
    VOC_CLASSES = ["FRONTEND","DOOREND","UPEND","slide","bromine_tank"]
    CreateYearsDarknetVocDatasets(r"F:\数据集\VOC数据集\箱门检测数据集\ContainVOC",  save_path=r"E:\Data\VOC数据集\箱门检测数据集\ContainerVOCDarknet",VOC_CLASSES=VOC_CLASSES)


if __name__ == '__main__':
    testCreateYearsDatasets()
