#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : jade_rename_voc_datasets.py
# @Author   : jade
# @Date     : 2021/11/30 17:32
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from jade import CreateSavePath,GetAllImagesNames
import os
from dataset_tools import DIRECTORY_IMAGES,DIRECTORY_ANNOTATIONS
import shutil
def RenameVOCDataSets(voc_path,save_path):
    CreateSavePath(save_path)
    save_jpg_path = CreateSavePath(os.path.join(save_path,DIRECTORY_IMAGES))
    save_anno_path = CreateSavePath(os.path.join(save_path,DIRECTORY_ANNOTATIONS))
    jpg_path = os.path.join(voc_path,DIRECTORY_IMAGES)
    anno_path  = os.path.join(voc_path,DIRECTORY_ANNOTATIONS)

    image_path_list = GetAllImagesNames(jpg_path)

    for image_path in image_path_list:
        file_name = image_path.split(".")[0]
        shutil.



if __name__ == '__main__':
    RenameVOCDataSets("","")

