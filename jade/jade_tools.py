#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : jade_tools.py
# @Author   : jade
# @Date     : 2021/5/25 10:28
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
import os
import sys
import datetime
import time
import shutil
from jade.jade_progress_bar import ProgressBar

def zh_ch(string):
    """
    解决cv2.namedWindow中文乱码问题
    :param string:
    :return:
    """
    return string.encode("gbk").decode('UTF-8', errors='ignore')

def getNumberofString(string):
    """
    提取字符串中的数字
    :param string:
    :return:
    """
    return "".join(list(filter(str.isdigit, string)))


def CreateSavePath(file_dir):
    """
    新建文件夹
    :param string:
    :return:返回文件夹名称
    """
    if os.path.exists(file_dir) is not True:
        os.makedirs(file_dir)
    return file_dir


def JudgeWhetherIPAddress(ip):
    """
    判断是否为IP地址
    :param string:
    :return:
    """
    if len(ip.split(".")) == 4:
        return True
    else:
        return False

def getConfig(config,section, configname):
    """
    读取ini参数
    """
    try:
        configparam = (config.get(section, configname)).split("#")[0].rstrip()
    except:
        print("读取{}参数异常,请检查参数是否正常".format(configname))
        sys.exit()
    return configparam

def getBoolConfig(config, section, configname):
    """
    读取ini参数,强制返回Bool
    :param string:
    :return:
    """
    try:
        configparam = (config.get(section, configname)).split("#")[0].rstrip()
    except:
        print("读取{}参数异常,请检查参数是否正常".format(configname))
        sys.exit()
    if configparam == "False":
        return False
    elif configparam == "True":
        return True
    else:
        print("读取{}参数异常,参数内容为:{}错误,请检查参数是否正常".format(configname,configparam))
        sys.exit()


def resource_path(relative_path):
    """
    生成资源文件目录访问路径,用于打包成可执行文件
    :param string:
    :return:
    """
    if getattr(sys, 'frozen', False): #是否Bundle Resource
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def GetSeqNumber():
    """
    返回序列号,精确到s
    :param string:
    :return:
    """
    now = datetime.datetime.now()
    otherStyleTime = now.strftime("%Y%m%d%H%M%S%f")
    return otherStyleTime


def timestr_to_time(time_str):
    """
    字符串转时间
    :param string:
    :return: Int
    """

    return time.mktime(time.strptime(time_str,"%Y-%m-%d %H:%M:%S"))



def timefloat_to_timestr(floatstring):
    """
    时间字符串戳转时间字符串
    :param string:
    :return: string
    """
    floattime = float(floatstring)
    timeArray = time.localtime(floattime)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime


def timeint_to_timestr(timeInt):
    """
    时间戳转时间字符串
    :param Int:
    :return: string
    """
    timeArray = time.localtime(timeInt)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime


def GetChineseTimeStamp():
    now = datetime.datetime.now()
    otherStyleTime = now.strftime("%Y{y}%m{m}%d{d} %H:%M:%S").format(y='年',m='月',d='日')
    pathname = otherStyleTime
    return pathname

def GetTimeStamp():
    now = datetime.datetime.now()
    otherStyleTime = now.strftime("%Y-%m-%d %H:%M:%S")
    pathname = otherStyleTime
    return pathname

def GetTime():
    now = datetime.datetime.now()
    otherStyleTime = now.strftime("%Y-%m-%d-%H-%M-%S")
    pathname = otherStyleTime
    data_ms = datetime.datetime.now().microsecond / 1000
    time_stamp = "%s-%03d" % (pathname, data_ms)
    return time_stamp

#合并文件路径
def OpsJoin(path1,path2):
    return os.path.join(path1,path2)

#返回上一层目录
def GetPreviousDir(savepath):

    return os.path.dirname(savepath)
#返回最后一层的目录
def GetLastDir(savepath):
    return os.path.basename(savepath)


#获取文件夹下，后缀为.的文件
def GetFilesWithLastNamePath(dir,lastname):
    imagename_list = os.listdir(dir)
    image_list = []
    for image_name in imagename_list:
        last = "."+image_name.split(".")[-1]
        if last == lastname:
            image_list.append(os.path.join(dir,image_name))
    return (image_list)

#获取一个文件夹下所有的图片列表
def GetAllImagesNames(dir):
    imagename_list = os.listdir(dir)
    image_list = []
    for image_name in imagename_list:
        if image_name[-4:].lower == ".jpg" or image_name[-4:].lower() == ".png":
            image_list.append(image_name)
    return (image_list)

#获取一个文件夹下所有的图片路径
def GetAllImagesPath(dir):
    imagename_list = os.listdir(dir)
    image_list = []
    for image_name in imagename_list:
        if image_name[-4:].lower() == ".jpg" or image_name[-4:].lower() == ".png":
            image_list.append(OpsJoin(dir,image_name))
    return (image_list)

#获取今天的日期
def GetToday():
    now = datetime.datetime.now()
    otherStyleTime = now.strftime("%Y-%m-%d %H:%M:%S")
    pathname = otherStyleTime.split(" ")[0]
    return pathname

#获取当前的时间
def GetHourTime():
    now = datetime.datetime.now()
    otherStyleTime = now.strftime("%Y-%m-%d %H-%M-%S")
    pathname = otherStyleTime.split(" ")[1]
    return pathname



##文件夹下文件重新命名
def RenameImageWithDir(dir):
    image_path_list = GetAllImagesPath(dir)
    progressBar = ProgressBar(len(image_path_list))
    for image_path in image_path_list:
        shutil.copy(image_path,os.path.join(dir,GetSeqNumber()+".jpg"))
        os.remove(image_path)
        progressBar.update()
if __name__ == '__main__':
    RenameImageWithDir(r"F:\现场数据\镇江大港\车牌图片\2021-12-15")


