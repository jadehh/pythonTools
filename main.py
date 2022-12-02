#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : main.py
# @Author   : dataset_tools
# @Date     : 2021/5/6 9:38
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from dataset_tools.jade_create_paddle_text_detection_datasets import *
from dataset_tools.jade_create_paddle_ocr_datasets import *
from dataset_tools.jade_create_object_dection_datasets import CreateYearsDatasets
if __name__ == '__main__':
    #removeNolabelDatasets(r"F:\数据集\关键点检测数据集\定制版箱号关键点数据集\2022-03-09")
    #create_text_detection_datasets(r"F:\数据集\关键点检测数据集\定制版箱号关键点数据集",r"E:\Data\字符检测识别数据集\定制版箱号关键点数据集",0.95)
    #CreatePaddleOCRDatasets(root_path="E:\Data\字符检测识别数据集\镇江大港厂内车牌关键点检测数据集", save_path="E:\Data\OCR\镇江大港厂内车牌识别数据集",dataset_type="镇江厂内车牌数据集")
    #removeNolabelVocDatasets(r"E:\Data\VOC数据集\集装箱残损检测数据集")
    #CreateYearsDatasets(r"E:\Data\VOC数据集\集装箱残损检测数据集")
    #create_text_detection_datasets(r"F:\数据集\关键点检测数据集\箱号关键点数据集",r'E:\Data\字符检测识别数据集\箱号关键点数据集')
    #CreatePaddleOCRDatasets(r'F:\数据集\VOC数据集\箱门检测数据集\ContainVOC', save_path="E:\Data\OCR\箱号识别数据集",dataset_type="箱号数据集")
    CreateYearsDatasets("E:\Data\VOC数据集\箱门检测数据集\ContainVOC",0.95)