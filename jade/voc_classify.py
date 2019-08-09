#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/7/31 by jade
# 邮箱：jadehh@live.com
# 描述：tensorflow 2.0 keras 的使用
# 最近修改：2019/7/31  下午10:42 modify by jade

import os
from jade.ReadVocData import *
import cv2
def VOCTOClassify(VOC_PATH):
    xmlPaths = GetFilesWithLastNamePath(os.path.join(VOC_PATH,"Annotations",),".xml")
    save_path = GetPreviousDir(VOC_PATH)
    processbar = ProcessBar()
    processbar.count = len(xmlPaths)
    save_path = CreateSavePath(os.path.join(save_path,"Classify"))
    for xml_path in xmlPaths:
        imagename,shape, bboxes, labels_text,labels, difficult, truncated = ProcessXml(xml_path)
        image = cv2.imread(os.path.join(VOC_PATH,"JPEGImages",imagename))
        shape = image.shape
        for i in range(len(bboxes)):
            img = image[int(bboxes[i][1]*shape[0]):int(bboxes[i][3]*shape[0]),int(bboxes[i][0]*shape[1]):int(bboxes[i][2]*shape[1]),]
            CreateSavePath(os.path.join(save_path,labels_text[i]))
            cv2.imwrite(os.path.join(save_path,labels_text[i],imagename[:-4]+"_"+str(i)+".jpg"),img)
        NoLinePrint("writing images",processbar)
if __name__ == '__main__':
    voc_path = "/media/jade/CODE/Data/VOCdevkit/VOC2012/"
    voc_to_classify(voc_path)