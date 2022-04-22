#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : video_to_voc_datasets.py
# @Author   : jade
# @Date     : 2022/3/7 16:21
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
import os
from jade import *
from opencv_tools import *
import cv2

class Video_To_Image_Datasets(object):
    def __init__(self,root_path,save_path,detector=None,fps=5):
        self.root_path = root_path
        self.detector = detector
        self.save_path = CreateSavePath(os.path.join(save_path,GetToday(),DIRECTORY_IMAGES))
        CreateSavePath(os.path.join(save_path,GetToday(),DIRECTORY_ANNOTATIONS))
        self.fps = fps
        super(Video_To_Image_Datasets, self).__init__()

    def run(self):
        video_list = GetFilesWithLastNamePath(self.root_path,".avi")
        processBar = ProgressBar(len(video_list))
        for video_path in video_list:
            capture = cv2.VideoCapture(video_path)
            index = 0
            while capture.isOpened():
                ret,frame = capture.read()
                if ret is False:
                    break
                if self.detector is None:
                    if index % self.fps == 0:
                        WriteChienePath(os.path.join(self.save_path,GetSeqNumber()+".jpg"),frame)
                    index = index + 1
            processBar.update()


if __name__ == '__main__':
    video_To_VOC_Datasets = Video_To_Image_Datasets(r'F:\视频数据集\箱号视频\广东佛山\2022-03-04\2022-03-04\top',r'F:\数据集\VOC数据集\定制版顶相机箱号检测数据集',None)
    video_To_VOC_Datasets.run()
