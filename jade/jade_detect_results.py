#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/9/3 by jade
# 邮箱：jadehh@live.com
# 描述：目标检测结果类
# 最近修改：2019/9/3  下午3:12 modify by jade

class DetectResultModel():
    def __init__(self,boxes,label_texts,labelIds,scores):
        self.boxes = boxes
        self.label_texts = label_texts
        self.label_ids = labelIds
        self.scores = scores
