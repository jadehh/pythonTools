#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : jade_create_object_dection_datasets.py
# @Author   : jade
# @Date     : 2021/11/30 16:48
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     : 制作目标检测数据集
from dataset_tools import *
import os
from jade import ProgressBar,GetLastDir
import shutil
import random
import xml.etree.ElementTree as ET

def ProcessXml(xml_path):
    # Read the XML annotation file.
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Image shape.
    size = root.find('size')
    roorname = root.find('filename').text
    shape = [(size.find('height').text),
             (size.find('width').text),
             (size.find('depth').text)]
    # Find annotations.
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        #label = (obj.find('bndbox')).find('name').text
        label = obj.find('name').text
        if label in VOC_LABELS.keys():
            labels.append(str(VOC_LABELS[label][0]))
        else:
            labels.append(1)
        labels_text.append(label)

        if obj.find('difficult'):
            difficult.append((obj.find('difficult').text))
        else:
            difficult.append('0')
        if obj.find('truncated'):
            truncated.append((obj.find('truncated').text))
        else:
            truncated.append('0')

        bbox = obj.find('bndbox')
        bboxes.append((float(bbox.find('xmin').text) / float(shape[1]) ,float(bbox.find('ymin').text) / float(shape[0]),float(bbox.find('xmax').text)/float(shape[1]),float(bbox.find('ymax').text)/float(shape[0])))
    imagename = GetLastDir(xml_path)[:-4]+'.jpg'
    return imagename,shape, bboxes, labels_text,labels, difficult, truncated

def CreateYearsDatasets(dir,rate=0.95):
    years = os.listdir(dir)
    if os.path.exists(os.path.join(dir,"train.txt")):
        os.remove(os.path.join(dir,"train.txt"))
    if os.path.exists(os.path.join(dir,"test.txt")):
        os.remove(os.path.join(dir,"test.txt"))
    progressBar1 = ProgressBar(len(years))
    with open(os.path.join(dir,"train.txt"),"w") as f1:
        for year in years:
            if os.path.isdir(os.path.join(dir, year)):
                if os.path.exists(os.path.join(dir,year,DIRECTORY_IMAGES)) and os.path.exists(os.path.join(dir,year,DIRECTORY_ANNOTATIONS)):
                    CreateVOCDataset(os.path.join(dir,year),year,rate)
                with open(os.path.join(dir,year,"ImageSets","Main","train.txt")) as f2:
                    for content in f2.read().split("\n")[:-1]:
                        f1.write(content+"\n")
            progressBar1.update()

    progressbar2 = ProgressBar(len(years))
    with open(os.path.join(dir, "test.txt"), "w") as f1:
        for year in years:
            if os.path.isdir(os.path.join(dir, year)):
                if os.path.exists(os.path.join(dir,year,DIRECTORY_IMAGES)) and os.path.exists(os.path.join(dir,year,DIRECTORY_ANNOTATIONS)):
                    CreateVOCDataset(os.path.join(dir,year),year,rate)
                with open(os.path.join(dir, year, "ImageSets", "Main", "test.txt")) as f2:
                    for content in f2.read().split("\n")[:-1]:

                        f1.write(content + "\n")
            progressbar2.update()
    CreateLabelList(dir)


##制作VOC数据集
def CreateVOCDataset(dir, datasetname,rate=0.95):
    """

    :param dir:
    :param datasetname:
    :param rate:
    :return:
    """
    root_path = dir
    dataset_name = datasetname
    Annotations = DIRECTORY_ANNOTATIONS
    JPEGImages = DIRECTORY_IMAGES

    if os.path.exists(os.path.join(root_path, "ImageSets", "Main")) is not True:
        os.makedirs(os.path.join(root_path, "ImageSets", "Main"))
    else:
        shutil.rmtree(os.path.join(root_path, "ImageSets", "Main"))
        os.makedirs(os.path.join(root_path, "ImageSets", "Main"))
    Main_path = os.path.join(root_path, "ImageSets", "Main")
    image_files = os.listdir(os.path.join(root_path, JPEGImages))

    train_image_files = random.sample(image_files, int(len(image_files) *rate))
    test_image_files = [file for file in image_files if file not in train_image_files]

    for train_image_file in train_image_files:
        with open(os.path.join(Main_path, "train_var.txt"), "a") as f:
            # with open(os.path.join(Main_path, "train.txt"), "a") as f:
            image_file = dataset_name + "/" + JPEGImages + "/" + train_image_file
            xml_file = dataset_name + "/" + Annotations + "/" + train_image_file[:-4] + ".xml"
            filename = train_image_file[:-4]
            with open(os.path.join(dir,JPEGImages,train_image_file),"rb") as f2:
                if len(f2.read()) == 0:
                    pass
                else:
                    f.write(filename + "\n")
            # f.write(image_file + " " + xml_file + "\n")

    for test_image_file in test_image_files:
        with open(os.path.join(Main_path, "test_var.txt"), "a") as f:
            # with open(os.path.join(Main_path, "test.txt"), "a") as f:
            image_file = dataset_name + "/" + JPEGImages + "/" + test_image_file
            xml_file = dataset_name + "/" + Annotations + "/" + test_image_file[:-4] + ".xml"
            filename = test_image_file[:-4]
            with open(os.path.join(dir,JPEGImages,test_image_file),"rb") as f2:
                if len(f2.read()) == 0:
                    pass
                else:
                    f.write(filename + "\n")
            # f.write(image_file + " " + xml_file + "\n")

    for train_image_file in train_image_files:
        with open(os.path.join(Main_path, "train.txt"), "a") as f:
            # with open(os.path.join(Main_path, "train.txt"), "a") as f:
            image_file = dataset_name + "/" + JPEGImages + "/" + train_image_file
            xml_file = dataset_name + "/" + Annotations + "/" + train_image_file[:-4] + ".xml"
            filename = train_image_file[:-4]
            with open(os.path.join(dir,JPEGImages,train_image_file),"rb") as f2:
                if len(f2.read()) == 0:
                    pass
                else:
                    f.write(image_file + " " + xml_file + "\n")
            # f.write(filename + "\n")


    for test_image_file in test_image_files:
        with open(os.path.join(Main_path, "test.txt"), "a") as f:
            # with open(os.path.join(Main_path, "test.txt"), "a") as f:
            image_file = dataset_name + "/" + JPEGImages + "/" + test_image_file
            xml_file = dataset_name + "/" + Annotations + "/" + test_image_file[:-4] + ".xml"
            filename = test_image_file[:-4]
            with open(os.path.join(dir,JPEGImages,test_image_file),"rb") as f2:
                if len(f2.read()) == 0:
                    pass
                else:
                    f.write(image_file + " " + xml_file + "\n")
            # f.write(filename + "\n")

def CreateLabelList(dir):
    """
    :param dir:
    :return:
    """
    label_list = []
    if os.path.exists(os.path.join(dir,"train.txt")):
        with open(os.path.join(dir,"train.txt"),"r") as f:
            content_list= f.read().split("\n")[:-1]
            progressbar = ProgressBar(len(content_list))
            for content in content_list:
                ann_file_path = os.path.join(dir,content.split(" ")[-1])
                imagename, shape, bboxes, labels_text, labels, difficult, truncated = ProcessXml(ann_file_path)
                for label in labels_text:
                    if label not in label_list:
                        label_list.append(label)
                progressbar.update()
        progressBar2 = ProgressBar(len(label_list))
        for label_name in label_list:
            with open(os.path.join(dir,"label_list.txt"),"a") as f:
                f.write(label_name+"\n")
            progressBar2.update()
    else:
        print("请先制作数据集")

if __name__ == '__main__':
    print("Done")