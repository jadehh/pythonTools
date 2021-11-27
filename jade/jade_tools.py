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

def getSeqNumber():
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
    return ops.join(path1,path2)

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

class ProgressBar:
    """A progress bar which can print the progress.终端进度条"""

    def __init__(self, task_num=0, bar_width=50, start=True, file=sys.stdout):
        self.task_num = task_num
        self.bar_width = bar_width
        self.completed = 0
        self.file = file
        if start:
            self.start()

    @property
    def terminal_width(self):
        width, _ = get_terminal_size()
        return width

    def start(self):
        if self.task_num > 0:
            self.file.write(f'[{" " * self.bar_width}] 0/{self.task_num}, '
                            '花费了: 0s, 预计还剩:')
        else:
            self.file.write('完成: 0, 共花费: 0s')
        self.file.flush()
        self.timer = Timer()

    def update(self, num_tasks=1):
        assert num_tasks > 0
        self.completed += num_tasks
        elapsed = self.timer.since_start()
        if elapsed > 0:
            fps = self.completed / elapsed
        else:
            fps = float('inf')
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            msg = f'\r[{{}}] {self.completed}/{self.task_num}, ' \
                  f'{fps:.1f} task/s, 花费了: {int(elapsed + 0.5)}s, ' \
                  f'预计还剩: {eta:5}s'

            bar_width = min(self.bar_width,
                            int(self.terminal_width - len(msg)) + 2,
                            int(self.terminal_width * 0.6))
            bar_width = max(2, bar_width)
            mark_width = int(bar_width * percentage)
            bar_chars = '>' * mark_width + ' ' * (bar_width - mark_width)
            self.file.write(msg.format(bar_chars))
        else:
            self.file.write(
                f'完成: {self.completed}, 共花费: {int(elapsed + 0.5)}s,'
                f' {fps:.1f} tasks/s')
        self.file.flush()


