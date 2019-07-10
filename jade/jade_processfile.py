#coding=utf-8
import os
import os.path as ops
from jade import DIRECTORY_IMAGES,DIRECTORY_ANNOTATIONS,GetAllImagesPath,OpsJoin



def GetVOCTrainImagePath(dir):
    "/home/jade/Data/StaticDeepFreeze/wild_goods_voc///"
    with open(os.path.join(dir,"ImageSets","Main","train_var.txt"),'r') as f:
        result = f.read().split("\n")[:-1]

    train = []
    for train_name in result:
        train.append(os.path.join(dir,DIRECTORY_IMAGES,train_name.split(" ")[0] + ".jpg"))
    return train


def GetVOCTrainXmlPath(dir):
    "/home/jade/Data/StaticDeepFreeze/wild_goods_voc///"
    with open(os.path.join(dir,"ImageSets","Main","train_var.txt"),'r') as f:
        result = f.read().split("\n")[:-1]

    train = []
    for train_name in result:
        train.append(os.path.join(dir,DIRECTORY_ANNOTATIONS,train_name.split(" ")[0] + ".xml"))
    return train


def GetVOCTestImagePath(dir):
    "/home/jade/Data/StaticDeepFreeze/wild_goods_voc///"
    with open(os.path.join(dir,"ImageSets","Main","test_var.txt"),'r') as f:
        result = f.read().split("\n")[:-1]

    test = []
    for test_name in result:
        test.append(os.path.join(dir,DIRECTORY_IMAGES,test_name + ".jpg"))
    return test

def GetVOCTrainXmlPath(dir):
    "/home/jade/Data/StaticDeepFreeze/wild_goods_voc///"
    with open(os.path.join(dir,"ImageSets","Main","train_var.txt"),'r') as f:
        result = f.read().split("\n")[:-1]

    test = []
    for test_name in result:
        test.append(os.path.join(dir,DIRECTORY_ANNOTATIONS,test_name + ".xml"))
    return test

def GetVOCTestXmlPath(dir):
    "/home/jade/Data/StaticDeepFreeze/wild_goods_voc///"
    with open(os.path.join(dir,"ImageSets","Main","test_var.txt"),'r') as f:
        result = f.read().split("\n")[:-1]

    test = []
    for test_name in result:
        test.append(os.path.join(dir,DIRECTORY_ANNOTATIONS,test_name + ".xml"))
    return test

#获取标签文件夹下所有的图片路径和标签名
def GetLabelAndImages(dir):
    labels = os.listdir(dir)
    image_list = []
    labels_n = []
    _CLASS_NAMES = [
        'hand_no_goods',
        'hand_in_goods',
    ]
    for label in labels:
        imagename_list = os.listdir(os.path.join(dir,label))
        for image_name in imagename_list:
            if image_name[-4:] == ".jpg" or image_name[-4:] == ".png" or image_name[-5:] == ".jpeg":
                image_list.append(os.path.join(os.path.join(dir,label), image_name))
                labels_n.append(int(_CLASS_NAMES.index(label)))
    return image_list, labels_n

#获取根目录


#新建目录
def CreateSavePath(save_image_path):
    if os.path.exists(save_image_path) is not True:
        os.makedirs(save_image_path)
    return save_image_path

#返回JPEGTImages VOC
def XMLPathTOImagePath(xml_path):
    paths = xml_path.split(DIRECTORY_ANNOTATIONS)
    return OpsJoin(paths[0],DIRECTORY_IMAGES,GetLastDir(xml_path)[:-4]+".jpg")

def GetVOCImageDir(root_path):
    return GetAllImagesPath(OpsJoin(root_path,DIRECTORY_IMAGES))

def GetVOCDir(image_path):
    paths = image_path.split(DIRECTORY_IMAGES)
    return paths[0]

def GetVOCXmlDir(dir):
    imagename_list = os.listdir(os.path.join(dir,DIRECTORY_ANNOTATIONS))
    image_list = []
    for image_name in imagename_list:
        if image_name[-4:] == ".xml":
            image_list.append(os.path.join(os.path.join(dir,DIRECTORY_ANNOTATIONS),image_name))
    return (image_list)

def GetVOCCombineImageDir(dir):
    imagename_list = os.listdir(os.path.join(dir,"Combine_Image_XML"))
    image_list = []
    for image_name in imagename_list:
        if image_name[-4:] == ".jpg" or image_name[-4:] == ".png" or image_name[-5:]==".jpeg":
            image_list.append(os.path.join(os.path.join(dir,"Combine_Image_XML"),image_name))
    return (image_list)

#新建VOC目录
def CreateVOCSavePath(save_path):
    if os.path.exists(os.path.join(save_path,DIRECTORY_IMAGES)) is not True:
        os.makedirs(os.path.join(save_path,DIRECTORY_IMAGES))
    if os.path.exists(os.path.join(save_path,DIRECTORY_ANNOTATIONS)) is not True:
        os.makedirs(os.path.join(save_path,DIRECTORY_ANNOTATIONS))


