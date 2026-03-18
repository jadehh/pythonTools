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
from dataset_tools.jade_voc_datasets import GetXmlClassesNames,GenerateXml


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

def CreateYearsDatasets(dir,year=None,save_path=None,rate=0.95,remove_classes=None):
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
                    CreateVOCDataset(os.path.join(dir, year), year, save_path, rate,remove_classes)
            progressBar1.update()
    else:
        if os.path.isdir(os.path.join(dir, year)):
            if os.path.exists(os.path.join(dir, year, DIRECTORY_IMAGES)) and os.path.exists(
                    os.path.join(dir, year, DIRECTORY_ANNOTATIONS)):
                CreateVOCDataset(os.path.join(dir, year), year, save_path, rate,remove_classes)

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


# VOC 数据集转换为Darknet数据集
def CreateYearsDarknetVocDatasets(dir, year=None, save_path=None, rate=0.95,VOC_CLASSES=None):
    years = os.listdir(dir)
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
                    CreateDarknetVocDatasets(os.path.join(dir, year), save_path, rate,VOC_CLASSES)
            progressBar1.update()
    else:
        if os.path.isdir(os.path.join(dir, year)):
            if os.path.exists(os.path.join(dir, year, DIRECTORY_IMAGES)) and os.path.exists(os.path.join(dir, year, DIRECTORY_ANNOTATIONS)):
                CreateDarknetVocDatasets(os.path.join(dir, year), save_path, rate,VOC_CLASSES)
        progressBar1.update()

    with open(os.path.join(save_path,"classes.txt"),"wb") as f:
        for class_name in VOC_CLASSES:
            f.write((class_name+"\n").encode("utf-8"))


def convert_voc_to_yolo(xml_dir, output_dir, classes):
    tree = ET.parse(xml_dir)
    root = tree.getroot()
    img_w = int(root.find('size/width').text)
    img_h = int(root.find('size/height').text)

    with open(os.path.join(output_dir), 'w') as f:
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            cls_id = classes.index(cls_name)
            bbox = obj.find('bndbox')
            x_center = (int(bbox.find('xmin').text) + int(bbox.find('xmax').text)) / 2 / img_w
            y_center = (int(bbox.find('ymin').text) + int(bbox.find('ymax').text)) / 2 / img_h
            width = (int(bbox.find('xmax').text) - int(bbox.find('xmin').text)) / img_w
            height = (int(bbox.find('ymax').text) - int(bbox.find('ymin').text)) / img_h
            f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")

def CreateDarknetVocDataset(dir,save_path,image_files,dataset_type,remove_label="None",VOC_CLASSES=None):
    save_image_path = CreateSavePath(os.path.join(save_path,"images",dataset_type,))
    save_label_path = CreateSavePath(os.path.join(save_path,"labels",dataset_type))
    for image_file in image_files:
        with open(os.path.join(dir, DIRECTORY_IMAGES, image_file), "rb") as f2:
            if len(f2.read()) == 0:
                pass
            else:
                class_name_list = GetXmlClassesNames(os.path.join(dir, DIRECTORY_ANNOTATIONS, image_file[:-4] + ".xml"))
                if len(class_name_list) > 0 and remove_label not in class_name_list:
                    shutil.copy(os.path.join(dir, DIRECTORY_IMAGES, image_file), save_image_path)
                    convert_voc_to_yolo(os.path.join(dir,DIRECTORY_ANNOTATIONS,image_file[:-4] + ".xml"),os.path.join(save_label_path,image_file[:-4] + ".txt"),VOC_CLASSES)
                else:
                    print("未找到类别:{}".format(os.path.join(dir, DIRECTORY_ANNOTATIONS, image_file[:-4] + ".xml")))




def CreateDarknetVocDatasets(dir,save_path,rate,VOC_CLASSES):
    """
    :param dir:
    """
    image_files = os.listdir(os.path.join(dir, DIRECTORY_IMAGES))
    train_image_files = random.sample(image_files, int(len(image_files) *rate))
    test_image_files = [file for file in image_files if file not in train_image_files]
    CreateDarknetVocDataset(dir,save_path,train_image_files,"train",VOC_CLASSES=VOC_CLASSES)
    CreateDarknetVocDataset(dir,save_path,test_image_files,"test",VOC_CLASSES=VOC_CLASSES)


def generate_new_datasets(dir,dataset_name,root_path,image_file_list,no_pretrained_images_dir,output,remove_classes=None):
    Main_path = os.path.join(root_path, "ImageSets", "Main")
    for image_file in image_file_list:
        is_success = False
        img_file = dataset_name + "/" + DIRECTORY_IMAGES + "/" + image_file
        xml_file = dataset_name + "/" + DIRECTORY_ANNOTATIONS + "/" + image_file[:-4] + ".xml"
        is_remove = False
        with open(os.path.join(Main_path, output+".txt"), "a") as f:
            # with open(os.path.join(Main_path, "train.txt"), "a") as f:
            save_image_path = CreateSavePath(os.path.join(root_path,DIRECTORY_IMAGES))
            save_xml_path = CreateSavePath(os.path.join(root_path,DIRECTORY_ANNOTATIONS))
            with open(os.path.join(dir,DIRECTORY_IMAGES,image_file),"rb") as f2:
                if len(f2.read()) == 0 and ReadChinesePath(os.path.join(dir,DIRECTORY_IMAGES,image_file)) != None:
                    pass
                else:
                    imagename,shape, bboxes, labels_text,labels, difficult, truncated = ProcessXml(os.path.join(dir, DIRECTORY_ANNOTATIONS, image_file[:-4] + ".xml"))
                    for class_name in labels_text:
                        if class_name in remove_classes:
                            is_remove = True
                    if len(labels) > 0:
                        if is_remove:
                            print("删除类别:{}".format(os.path.join(dir, DIRECTORY_ANNOTATIONS, image_file[:-4] + ".xml")))
                            GenerateXml(image_file[:-4] ,shape,[],[], save_xml_path)
                        else:
                            shutil.copy(os.path.join(dir, DIRECTORY_ANNOTATIONS, image_file[:-4] + ".xml"),save_xml_path)
                        shutil.copy(os.path.join(dir, DIRECTORY_IMAGES, image_file), save_image_path)
                        is_success = True
                        f.write(img_file + " " + xml_file + "\n")
                    else:
                        print("未找到类别:{}".format(os.path.join(dir, DIRECTORY_ANNOTATIONS, image_file[:-4] + ".xml")))
        if is_success is False:
            shutil.copy(os.path.join(dir, DIRECTORY_IMAGES, image_file),os.path.join(no_pretrained_images_dir, image_file))
            try:
                os.remove(os.path.join(dir, DIRECTORY_IMAGES, image_file))
                os.remove(os.path.join(dir, DIRECTORY_ANNOTATIONS, image_file[:-4] + ".xml"))
            except Exception as e:
                print("删除失败,{}".format(e))
                pass
            print("未找到类别:{}".format(os.path.join(dir, DIRECTORY_ANNOTATIONS, image_file[:-4] + ".xml")))
        shutil.copy(os.path.join(Main_path, output+".txt"), os.path.join(Main_path, output+"_var.txt"))


##制作VOC数据集
def CreateVOCDataset(dir, datasetname,save_path=None,rate=0.95,remove_classes=None):
    """

    :param dir:
    :param datasetname:
    :param rate:
    :return:
    """
    no_pretrained_dir =  CreateSavePath(os.path.join(os.path.dirname(dir),"no_pretrained"))

    root_path = os.path.join(save_path,datasetname)
    dataset_name = datasetname
    Annotations = DIRECTORY_ANNOTATIONS
    JPEGImages = DIRECTORY_IMAGES
    no_pretrained_images_dir = CreateSavePath(os.path.join(no_pretrained_dir,JPEGImages))

    if os.path.exists(os.path.join(root_path, "ImageSets", "Main")) is not True:
        os.makedirs(os.path.join(root_path, "ImageSets", "Main"))
    else:
        shutil.rmtree(os.path.join(root_path, "ImageSets", "Main"))
        os.makedirs(os.path.join(root_path, "ImageSets", "Main"))
    image_files = os.listdir(os.path.join(dir, JPEGImages))
    train_image_files = random.sample(image_files, int(len(image_files) *rate))
    test_image_files = [file for file in image_files if file not in train_image_files]

    generate_new_datasets(dir,dataset_name,root_path,train_image_files,no_pretrained_images_dir,"train",remove_classes)
    generate_new_datasets(dir,dataset_name,root_path,test_image_files,no_pretrained_images_dir,"test",remove_classes)


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