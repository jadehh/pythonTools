#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/7/31 by jade
# 邮箱：jadehh@live.com
# 描述：tensorflow 2.0 keras 的使用
# 最近修改：2019/7/31  下午10:42 modify by jade

import os
from jade.ReadVocData import *
import cv2
def voc_to_classify(VOC_PATH):
    xmlPaths = GetFilesWithLastNamePath(os.path.join(VOC_PATH,"Annotations",),".xml")
    for xml_path in xmlPaths:
        imagename,shape, bboxes, labels_text,labels, difficult, truncated = ProcessXml(xml_path)
        image = cv2.imread(os.path.join(VOC_PATH,"JPEGImages",imagename))
        shape = image.shape
        for i in range(len(bboxes)):
            img = image[bboxes[i][0]:bboxes[i][2],bboxes[i][1]:bboxes[i][3],]
            cv2.imshow("result",img)
            cv2.waitKey(0)

if __name__ == '__main__':
    voc_path = "/media/jade/CODE/Data/VOCdevkit/VOC2012/"
    voc_to_classify(voc_path)