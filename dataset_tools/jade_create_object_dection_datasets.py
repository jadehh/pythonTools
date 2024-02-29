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
from jade import ProgressBar,GetLastDir,CreateSavePath
import shutil
import random
import xml.etree.ElementTree as ET

def CreateSavePath(save_path):
    if os.path.exists(save_path):
        return save_path
    else:
        os.makedirs(save_path)
        return save_path

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

def CreateYearsDatasets(dir,year=None,save_path=None,rate=0.95):
    years = os.listdir(dir)
    if os.path.exists(os.path.join(save_path,"train.txt")):
        os.remove(os.path.join(save_path,"train.txt"))
    if os.path.exists(os.path.join(save_path,"test.txt")):
        os.remove(os.path.join(save_path,"test.txt"))
    if year is None:
        progressBar1 = ProgressBar(len(years))
    else:
        progressBar1 = ProgressBar(1)
    if os.path.exists(save_path):
        pass
    else:
        os.makedirs(save_path)
    if year is None:
        for year in years:
            if os.path.isdir(os.path.join(dir, year)):
                if os.path.exists(os.path.join(dir, year, DIRECTORY_IMAGES)) and os.path.exists(
                        os.path.join(dir, year, DIRECTORY_ANNOTATIONS)):
                    CreateVOCDataset(os.path.join(dir, year), year, save_path, rate)
            progressBar1.update()
    else:
        if os.path.isdir(os.path.join(dir, year)):
            if os.path.exists(os.path.join(dir, year, DIRECTORY_IMAGES)) and os.path.exists(
                    os.path.join(dir, year, DIRECTORY_ANNOTATIONS)):
                CreateVOCDataset(os.path.join(dir, year), year, save_path, rate)

    years = os.listdir(save_path)
    with open(os.path.join(save_path, "train.txt"), "w") as f1:
        progressbar2 = ProgressBar(len(years))
        for year in years:
            if os.path.isdir(os.path.join(dir, year)):
                with open(os.path.join(save_path, year, "ImageSets", "Main", "train.txt")) as f2:
                    for content in f2.read().split("\n")[:-1]:
                        f1.write(content + "\n")
        progressbar2.update()

    with open(os.path.join(save_path, "test.txt"), "w") as f1:
        progressbar2 = ProgressBar(len(years))
        for year in years:
            if os.path.isdir(os.path.join(dir, year)):
                with open(os.path.join(save_path, year, "ImageSets", "Main", "test.txt")) as f2:
                    for content in f2.read().split("\n")[:-1]:
                        f1.write(content + "\n")
        progressbar2.update()
    CreateLabelList(save_path)


##制作VOC数据集
def CreateVOCDataset(dir, datasetname,save_path=None,rate=0.95):
    """

    :param dir:
    :param datasetname:
    :param rate:
    :return:
    """
    root_path = os.path.join(save_path,datasetname)
    dataset_name = datasetname
    Annotations = DIRECTORY_ANNOTATIONS
    JPEGImages = DIRECTORY_IMAGES

    if os.path.exists(os.path.join(root_path, "ImageSets", "Main")) is not True:
        os.makedirs(os.path.join(root_path, "ImageSets", "Main"))
    else:
        shutil.rmtree(os.path.join(root_path, "ImageSets", "Main"))
        os.makedirs(os.path.join(root_path, "ImageSets", "Main"))
    Main_path = os.path.join(root_path, "ImageSets", "Main")
    image_files = os.listdir(os.path.join(dir, JPEGImages))
    train_image_files = random.sample(image_files, int(len(image_files) *rate))
    test_image_files = [file for file in image_files if file not in train_image_files]

    for train_image_file in train_image_files:
        with open(os.path.join(Main_path, "train_var.txt"), "a") as f:
            # with open(os.path.join(Main_path, "train.txt"), "a") as f:
            image_file = dataset_name + "/" + JPEGImages + "/" + train_image_file
            xml_file = dataset_name + "/" + Annotations + "/" + train_image_file[:-4] + ".xml"
            filename = train_image_file[:-4]
            save_image_path = CreateSavePath(os.path.join(root_path,DIRECTORY_IMAGES))
            save_xml_path = CreateSavePath(os.path.join(root_path,DIRECTORY_ANNOTATIONS))
            with open(os.path.join(dir,JPEGImages,train_image_file),"rb") as f2:
                if len(f2.read()) == 0:
                    pass
                else:
                    shutil.copy(os.path.join(dir, JPEGImages, train_image_file), save_image_path)
                    shutil.copy(os.path.join(dir, DIRECTORY_ANNOTATIONS, train_image_file[:-4] + ".xml"), save_xml_path)
                    f.write(filename + "\n")
            # f.write(image_file + " " + xml_file + "\n")

    for test_image_file in test_image_files:
        with open(os.path.join(Main_path, "test_var.txt"), "a") as f:
            # with open(os.path.join(Main_path, "test.txt"), "a") as f:
            image_file = dataset_name + "/" + JPEGImages + "/" + test_image_file
            xml_file = dataset_name + "/" + Annotations + "/" + test_image_file[:-4] + ".xml"
            filename = test_image_file[:-4]
            save_image_path = CreateSavePath(os.path.join(root_path, DIRECTORY_IMAGES))
            save_xml_path = CreateSavePath(os.path.join(root_path, DIRECTORY_ANNOTATIONS))
            with open(os.path.join(dir,JPEGImages,test_image_file),"rb") as f2:
                if len(f2.read()) == 0:
                    pass
                else:
                    shutil.copy(os.path.join(dir, JPEGImages, train_image_file), save_image_path)
                    shutil.copy(os.path.join(dir, DIRECTORY_ANNOTATIONS, train_image_file[:-4] + ".xml"), save_xml_path)
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
        with open(os.path.join(dir, "label_list.txt"), "wb") as f:
            for label_name in label_list:
                f.write((label_name+"\n").encode("utf-8"))
        progressBar2.update()
    else:
        print("请先制作数据集")

if __name__ == '__main__':
    print("Done")