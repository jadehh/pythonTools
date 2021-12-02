#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : jade_opencv_process.py
# @Author   : jade
# @Date     : 2021/11/30 10:00
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
##旋转图片
import cv2
import numpy as np
from jade import ProgressBar
import threading
import random
from PIL import Image, ImageFont, ImageDraw
import os
import time
import uuid
from opencv_tools import DIRECTORY_IMAGES,DIRECTORY_ANNOTATIONS,DIRECTORY_PREANNOTATIONS
import base64

##旋转图片
def Image_Roate(image, angle):
    # 获取图像的尺寸
    # 旋转中心
    (h, w) = image.shape[:2]
    (cx, cy) = (w / 2, h / 2)

    # 设置旋转矩阵
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算图像旋转后的新边界
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy

    return cv2.warpAffine(image, M, (nW, nH))


def Video_Roate(video_path, save_video_path, angle, fps=20):
    video_capture = cv2.VideoCapture(video_path)
    ret, frame = video_capture.read()
    roate_img = Image_Roate(frame, angle)
    height = roate_img.shape[0]
    width = roate_img.shape[1]
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter(save_video_path, fourcc, fps, (width, height))
    progressBar = ProgressBar(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        frame = Image_Roate(frame, angle)
        videoWriter.write(frame)
        ret, frame = video_capture.read()
        progressBar.update()
        if ret is not True:
            break


# 分割视频
def split_video(input_video_path, output_video_path, start_time, end_time):
    """

    :param input_video_path: 输入视频地址
    :param output_video_path: 输出视频地址
    :param start_time: 开始时间
    :param end_time: 结束时间
    :return:
    """
    video_capture = cv2.VideoCapture(input_video_path)
    ret, frame = video_capture.read()
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    height = frame.shape[0]
    width = frame.shape[1]
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    if int(fps) == 0:
        fps = 15
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    index = 0
    while True:
        ret, frame = video_capture.read()
        index = index + 1
        current_time = index / fps
        if current_time > start_time and current_time < end_time:
            video_writer.write(frame)
            print("Writing Videos")
        if current_time > end_time:
            break
        if ret is False:
            break


class VideoCapture():
    def __init__(self, cv_videocapture_param):
        self.cv_videocapture_param = cv_videocapture_param
        return

    def handle(self):
        # 开启线程，传入参数
        _thread = threading.Thread(target=self.open)
        _thread.setDaemon(True)
        _thread.start()  # 启动线程
        return

    def open(self):
        capture = cv2.VideoCapture(self.cv_videocapture_param)
        ret, frame = capture.read()
        while ret:
            print("正在开启线程读取视频")
            ret, frame = capture.read()


class VideoCaptureThread():

    def __init__(self, cv_videocapture_param):
        self.__init_param()
        self.open(cv_videocapture_param)
        return

    def __init_param(self):
        self.__cvCap = cv2.VideoCapture()
        self.__isOpened = False

        self.__cvMat = [[], [], []]
        self.__cvCap_read_index = 2
        self.__user_read_index = 0
        self.__ret = False

        self.__thread_brk = False
        self.__p_thread = None

        self.__is_discarding_frame = True

    def open(self, cv_videocapture_param):
        self.release()
        if (type(cv_videocapture_param) == str):
            self.__isOpened = self.__cvCap.open(int(cv_videocapture_param))
        else:
            self.__isOpened = self.__cvCap.open(cv_videocapture_param)
        self.__ret, self.__cvMat[0] = self.__cvCap.read()
        self.__cvMat[1] = self.__cvMat[0].copy()
        self.__cvMat[2] = self.__cvMat[0].copy()
        if (self.__isOpened):
            self.__thread_start()
        return self.__isOpened

    def __thread_start(self):
        self.__p_thread = threading.Thread(target=VideoCaptureThread.get_img_in_thread, args=(self,), daemon=True)
        self.__p_thread.start()
        print('have start thread')
        return

    def get_img_in_thread(self):
        while (self.__thread_brk is not True and self.__isOpened):
            self.__ret, self.__cvMat[self.__cvCap_read_index] = self.__cvCap.read()
            if (self.__ret):
                next_cvCap_read_index = self.__cvCap_read_index + 1
                if (next_cvCap_read_index == 3):
                    next_cvCap_read_index = 0
                if (next_cvCap_read_index != self.__user_read_index):
                    self.__cvCap_read_index = next_cvCap_read_index
                else:
                    self.__is_discarding_frame = True
            else:
                return

    def read(self):
        if (self.__isOpened is not True):
            print('Video open fail')
            return False, None
        else:
            ret = self.__ret
            image = None
            if (ret):
                image = self.__cvMat[self.__user_read_index].copy()
                next_user_read_index = self.__user_read_index + 1
                if (next_user_read_index == 3):
                    next_user_read_index = 0
                if (next_user_read_index != self.__cvCap_read_index):
                    self.__user_read_index = next_user_read_index
                else:
                    self.__is_discarding_frame = False
            return ret, image

    def __thread_release(self):
        self.__thread_brk = True
        if (self.__p_thread != None):
            self.__p_thread.join()
        return

    def release(self):
        self.__thread_release()
        self.__init_param()
        return

    def isOpened(self):
        return self.__isOpened

    def is_discarding_frame(self):
        return self.__is_discarding_frame

    def __del__(self):
        self.release()
        return


class processFile:
    def __init__(self, path):
        self.path = path

    # 去掉最后一个目录
    def Previous_Dir(self):
        paths = self.path.split("/")
        paths.remove(paths[len(paths) - 1])
        previous_dir = ""
        for path in paths:
            previous_dir = previous_dir + path + "/"
        return previous_dir


class processImage:
    def __init__(self, img):
        self.img = img

    # RGB转BGR
    def RGBTOBGR(self):
        return cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)

    # BGR转RGB，一般用于图像不能显示正常的颜色
    def BGRTORGB(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

    # 给图像画一个矩形框,默认为随机的颜色,bboxes为xmin/width
    def RECTANGLE(self, bboxes):
        shape = self.img.shape
        width = shape[1]
        height = shape[0]
        color_R = random.randint(1, 254)
        color_G = random.randint(1, 254)
        color_B = random.randint(1, 255)
        self.img = cv2.rectangle(self.img, (int(bboxes[1] * width), int(bboxes[0] * height)),
                                 (int(bboxes[2] * width), int(bboxes[3] * height)), (color_R, color_G, color_B), 2, 2)
        return self.img

    # 图像画点
    def CIRCLE(self, points):
        color_R = random.randint(1, 254)
        color_G = random.randint(1, 254)
        color_B = random.randint(1, 255)
        for point in points:
            self.img = cv2.circle(self.img, point, 2, (color_R, color_G, color_B), 2, 2)
        return self.img

    # 高斯去噪
    def Gaussian_Blur(self):
        blurred = cv2.GaussianBlur(self.img, (9, 9), 0)
        return blurred

    # 高斯去噪后阈值分割,返回彩色图像
    def Thresh_and_Blur(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(img, (9, 9), 0)
        (thresh_value, thresh) = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)
        for c in range(3):
            self.img[:, :, c] = np.where(thresh[:, :, ] == 0,
                                         0,
                                         self.img[:, :, c])
        return self.img

    # otsu阈值分割,返回彩色图
    def ThreshColor(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        (thresh_value, thresh) = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        for c in range(3):
            self.img[:, :, c] = np.where(thresh[:, :, ] == 0,
                                         0,
                                         self.img[:, :, c])
        return self.img

    # ousu阈值分割，灰度图
    def ThreshGray(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        (thresh_value, thresh) = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        return thresh

    # 图像倾斜矫正,一般是图像分割后做倾斜矫正
    def image_rectification(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh == 0))
        angle = cv2.minAreaRect(coords)[-1]
        # print(angle)
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        h = self.img.shape[0]
        w = self.img.shape[1]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(self.img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        target_coords = np.where(rotated[:, :, 1] > 0)
        if len(target_coords[0]) > 0:
            rotated = rotated[min(target_coords[0]):max(target_coords[0]), min(target_coords[1]):max(target_coords[1]),
                      :]
        # 需要裁剪边框
        return rotated

    # 添加中文label
    def add_chinese_label(self,font_path, label, pt1=(0, 0), color=(255, 0, 0), font_size=15):
        cv2img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
        pilimg = Image.fromarray(cv2img)
        # PIL图片上打印汉字
        draw = ImageDraw.Draw(pilimg)  # 图片上打印
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
        draw.text(pt1, label, color, font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
        cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
        return cv2charimg


class processVideo():
    def __init__(self, video_path):
        self.video_path = video_path

    def Video_To_Images(self, save_path, isthread=False, cut_fps=0):
        """
        :param save_path:
        :param isthread:
        :param cut_fps: 每几秒保存一次图片
        :return:
        """
        if type(self.video_path) == str:
            if isthread:
                video_capture = VideoCaptureThread(self.video_path)
            else:
                video_capture = cv2.VideoCapture(self.video_path)
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
            processbar = ProgressBar(count)
            ret, frame = video_capture.read()
            index = 0
            # 可以根据fps和视频长度的大小计算出一共有多少张图片

            while ret:
                processbar.start_time = time.time()
                index += 1
                if index % (cut_fps) == 0:
                    cv2.imwrite(os.path.join(save_path, DIRECTORY_IMAGES, str(uuid.uuid1()) + ".jpg"), frame)
                ret, frame = video_capture.read()
                processbar.update()
            video_capture.release()
        else:

            video_path1 = self.video_path[0]
            video_path2 = self.video_path[1]
            if isthread:
                video_capture1 = VideoCaptureThread(video_path1)
                video_capture2 = VideoCaptureThread(video_path2)
            else:
                video_capture1 = cv2.VideoCapture(video_path1)
                video_capture2 = cv2.VideoCapture(video_path1)

            fps = video_capture1.get(cv2.CAP_PROP_FPS)
            count = video_capture1.get(cv2.CAP_PROP_FRAME_COUNT)
            progressBar = ProgressBar(count)
            ret1, frame1 = video_capture1.read()
            ret2, frame2 = video_capture2.read()
            index = 0
            # 可以根据fps和视频长度的大小计算出一共有多少张图片
            while ret1:
                index += 1
                if index % (cut_fps) == 0:
                    savename = str(uuid.uuid1()) + ".jpg"
                    cv2.imwrite(os.path.join(save_path[0], DIRECTORY_IMAGES, savename), frame1)
                    cv2.imwrite(os.path.join(save_path[1], DIRECTORY_IMAGES, savename), frame2)
                ret1, frame1 = video_capture1.read()
                ret2, frame2 = video_capture2.read()
                progressBar.update()
            video_capture1.release()
            video_capture2.release()


# opencv 读取中文
def ReadChinesePath(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    return cv_img


# 随机出一个颜色
def GetRandomColor():
    r1 = random.randint(0, 255)
    r2 = random.randint(0, 255)
    r3 = random.randint(0, 255)
    return (r1, r2, r3)


# 图片加标题，有黑边
def Add_Title_Image(image, title):
    image_shape = image.shape
    image1 = Image.new("RGB", (image_shape[1], image_shape[0]))
    image2 = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image2 = image2.resize((image_shape[1] - 5 * 2, image_shape[0] - 50 * 2), )
    # image.thumbnail(size)

    draw = ImageDraw.Draw(image1)

    # use a truetype font
    font = ImageFont.truetype("arial.ttf", 40)

    draw.text((100, 0), title, font=font)
    bw, bh = image1.size
    lw, lh = image2.size

    image1.paste(image2, (bw - lw, int((bh - lh) / 2)))

    img = cv2.cvtColor(np.asarray(image1), cv2.COLOR_RGB2BGR)
    return img


# 添加中文label
def Add_Chinese_Label(img,font_path, label, pt1=(0, 0), color=GetRandomColor(), font_size=24):
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    pilimg = Image.fromarray(cv2img)
    # PIL图片上打印汉字
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
    draw.text(pt1, label, (int(color[0]), int(color[1]), int(color[2])), font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
    cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    return cv2charimg

def draw_text_list(img, font_path,label_list, pt_list=[], color_list=[], font_size_list=[]):
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    pilimg = Image.fromarray(cv2img)
    # PIL图片上打印汉字
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    for (label, pt, color, font_size) in zip(label_list, pt_list, color_list, font_size_list):
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
        draw.text(pt, label, color, font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
    cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    return cv2charimg




def draw_text_det_res(img, dt_boxes, txts=None):
    if isinstance(img, str):
        src_im = cv2.imread(img)
    else:
        src_im = img.copy()
    if txts is None:
        txts = [None] * len(dt_boxes)
    for idx, (box, txt) in enumerate(zip(dt_boxes, txts)):
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        if txt:
            src_im = draw_text_list(src_im, [txt[0]], [(box[0, 0], box[3, 1])], [(0, 0, 255)], font_size_list=[54])
        cv2.polylines(src_im, [box], True, color=(0, 0, 255), thickness=2)
    return src_im



def opencv_to_base64(image):
    image_byte = cv2.imencode('.jpg', image)[1].tobytes()
    base64_str = str(base64.b64encode(image_byte), encoding='utf-8')
    return base64_str