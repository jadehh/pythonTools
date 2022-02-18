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
from dataset_tools import DIRECTORY_IMAGES,DIRECTORY_ANNOTATIONS,ProcessXml,GenerateXml
import shutil
from jade import *
def RenameVOCDataSets(voc_path,save_path,edit_class_name,replace_class_name):
    CreateSavePath(save_path)
    save_jpg_path = CreateSavePath(os.path.join(save_path,DIRECTORY_IMAGES))
    save_anno_path = CreateSavePath(os.path.join(save_path,DIRECTORY_ANNOTATIONS))
    jpg_path = os.path.join(voc_path,DIRECTORY_IMAGES)
    anno_path  = os.path.join(voc_path,DIRECTORY_ANNOTATIONS)
    image_path_list = GetAllImagesNames(jpg_path)
    progressBar = ProgressBar(len(image_path_list))

    for image_path in image_path_list:
        file_name = image_path.split(".")[0]
        imagename,shape, bboxes, labels_text,labels, difficult, truncated = ProcessXml(os.path.join(anno_path,file_name+".xml"),False)
        shutil.copy(os.path.join(jpg_path,image_path),os.path.join(save_jpg_path,image_path))
        labels_text_replace = []
        for label in labels_text:
            if label == edit_class_name:
                labels_text_replace.append(replace_class_name)
            else:
                labels_text_replace.append(label)
        GenerateXml(file_name,shape,bboxes,labels_text_replace,save_anno_path)
        progressBar.update()





if __name__ == '__main__':
    RenameVOCDataSets(r"F:\数据集\VOC数据集\镇江大港车辆二维码检测\2022-02-17",r"F:\数据集\VOC数据集\镇江大港车辆二维码检测\2022-02-18","UPEND","qr_code")

