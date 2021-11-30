#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : jade_voc_to_classify_datasets.py
# @Author   : jade
# @Date     : 2021/11/30 16:54
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from dataset_tools.jade_read_voc_datasets import *
import cv2
from jade import ProgressBar,GetFilesWithLastNamePath,GetPreviousDir
def VOCTOClassify(VOC_PATH):
    xmlPaths = GetFilesWithLastNamePath(os.path.join(VOC_PATH,"Annotations",),".xml")
    save_path = GetPreviousDir(VOC_PATH)
    progressBar = ProgressBar(len(xmlPaths))
    save_path = CreateSavePath(os.path.join(save_path,"Classify"))
    for xml_path in xmlPaths:
        imagename,shape, bboxes, labels_text,labels, difficult, truncated = ProcessXml(xml_path)
        image = cv2.imread(os.path.join(VOC_PATH,"JPEGImages",imagename))
        shape = image.shape
        for i in range(len(bboxes)):
            img = image[int(bboxes[i][1]*shape[0]):int(bboxes[i][3]*shape[0]),int(bboxes[i][0]*shape[1]):int(bboxes[i][2]*shape[1]),]
            height = img.shape[0]
            width = img.shape[1]
            if height > 224 and width > 224:
                CreateSavePath(os.path.join(save_path,labels_text[i]))
                cv2.imwrite(os.path.join(save_path,labels_text[i],imagename[:-4]+"_"+str(i)+".jpg"),img)
        progressBar.update()
if __name__ == '__main__':
    voc_path = "/home/jade/Data/VOCdevkit/VOC2012"
    VOCTOClassify(voc_path)
