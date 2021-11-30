#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : __init__.py.py
# @Author   : jade
# @Date     : 2021/11/30 16:48
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'JPEGImages/'
DIRECTORY_PREANNOTATIONS = "PredictAnnotations/"

VOC_LABELS = {'__background__':(0, 'Background'),  # always index 0
               'aeroplane':(1,'aeroplane'),
              'bicycle':(2,'bicycle'),
              'bird':(3,'bird'),
              'boat':(4,'boat'),
               'bottle':(5,'bottle'),
              'bus':(6,'bus'),
              'car':(7,'car'),
              'cat':(8,'cat'),
              'chair':(9,'chari'),
               'cow':(10,'cow'),
             'diningtable':(11,'diningtable'),
              'dog':(12,'dog'),
              'horse':(13,'horse'),
               'motorbike':(14,'motorbike'),
              'person':(15,'person'),
              'pottedplant':(16,'pottedplant'),
               'sheep':(17,'sheep'),
              'sofa':(18,'sofa'),
              'train':(19,'train'),
              'tvmonitor':(20,'tvmonitor')}

from dataset_tools.jade_create_object_dection_datasets import *
from dataset_tools.jade_read_voc_datasets import *
from dataset_tools.jade_voc_to_classify_datasets import *
from dataset_tools.jade_create_classify_dataset import *
from dataset_tools.coco_dataset_to_voc_dataset import *