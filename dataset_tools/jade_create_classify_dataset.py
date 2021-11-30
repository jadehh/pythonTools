#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/15 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/15  下午4:15 modify by jade
import os
import random
import shutil
from jade import ProgressBar
def RandomSplitImagesKeepBalance(classify_path):
    """
    为了保证数据集的平衡
    :param classify_path:
    :return:
    """
    save_dir = os.path.join(classify_path+"_Balance")
    if os.path.exists(save_dir):
        raise ValueError("数据已经平衡了")
    else:
        os.makedirs(save_dir)
    file_list = os.listdir(classify_path)
    min_sample_num = 0
    for i in range(len(file_list)):
        if min_sample_num < len(os.listdir(os.path.join(classify_path,file_list[i]))):
            min_sample_num = len(os.listdir(os.path.join(classify_path,file_list[i])))
    progressBar = ProgressBar(len(file_list))

    for i in range(len(file_list)):
        sample_list = os.listdir(os.path.join(classify_path,file_list[i]))
        random.shuffle(sample_list)
        for j in range(len(sample_list)):
            save_img_dir = os.path.join(save_dir,file_list[i])
            if os.path.exists(save_img_dir) is not True:
                os.makedirs(save_img_dir)
            shutil.copy(os.path.join(classify_path,file_list[i],sample_list[i]),os.path.join(save_dir,file_list[i],sample_list[i]))
        progressBar.update()
