#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : test.py
# @Author   : jade
# @Date     : 2024/2/27 9:16
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from dataset_tools.jade_create_object_dection_datasets import *
from opencv_tools import ReadChinesePath
import cv2


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

def image_to_video():
    image = ReadChinesePath(r"C:\Users\Administrator\Desktop\back_2025-05-07-15-40-44-701.jpg")
    height, width, _ = image.shape
    video_writer = cv2.VideoWriter(
        r"C:\Users\Administrator\Desktop\output_video.avi",
        cv2.VideoWriter_fourcc(*"XVID"),
        30,  # Frame rate
        (width, height)
    )

    index = 0
    while index < 1000:
        video_writer.write(image)
        index += 1

    video_writer.release()


if __name__ == '__main__':
    image_to_video()
