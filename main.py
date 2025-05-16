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

def test_create_paddle_years_datasets(args):
    CreateYearsDatasets(args.input_dataset_dir,None,save_path=args.save_dataset_dir,rate=0.9)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="制作数据集脚本")
    parser.add_argument("--dataset_type", default='paddle_detection', help="制作数据集的类型")
    parser.add_argument("--input_dataset_dir", default='test', help="数据集的地址")
    parser.add_argument("--save_dataset_dir", default='test/output_seals_01', help="保存数据集的地址")
    parser.add_argument("--voc_labels",  nargs='+',default="", help="类别")
    args = parser.parse_args()
    print(list(args.voc_labels))
    if args.dataset_type == "paddle_detection":
        CreateYearsDatasets(args.input_dataset_dir, None, save_path=args.save_dataset_dir, rate=0.9)
    elif args.dataset_type == "yolo_detection":
        CreateDarknetVocDatasets(args.input_dataset_dir,  save_path=args.save_dataset_dir, rate=0.9, VOC_CLASSES=args.voc_labels)

    #removeNolabelDatasets(r"F:\数据集\关键点检测数据集\定制版箱号关键点数据集\2022-03-09")
    #create_text_detection_datasets(r"F:\数据集\关键点检测数据集\定制版箱号关键点数据集",r"E:\Data\字符检测识别数据集\定制版箱号关键点数据集",0.95)
    #CreatePaddleOCRDatasets(root_path="E:\Data\字符检测识别数据集\镇江大港厂内车牌关键点检测数据集", save_path="E:\Data\OCR\镇江大港厂内车牌识别数据集",dataset_type="镇江厂内车牌数据集")
    #removeNolabelVocDatasets(r"E:\Data\VOC数据集\集装箱残损检测数据集")
    #CreateYearsDatasets(r"E:\Data\VOC数据集\集装箱残损检测数据集")
    #CreatePaddleOCRDatasets(r'F:\数据集\VOC数据集\箱门检测数据集\ContainVOC', save_path="E:\Data\OCR\箱号识别数据集",dataset_type="箱号数据集")
    #CreateYearsDatasets("F:\数据集\VOC数据集\验残集装箱检测数据集",0.95)