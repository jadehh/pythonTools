#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/22 by jade
# 邮箱：jadehh@live.com
# 描述：TODO 视频转码
# 最近修改：2019/8/22  上午9:59 modify by jade
import cv2
from jade import *
def videoToAvi(video_path):
    if GetLastDir(video_path)[-4:] != ".mp4":
        raise ValueError("现在仅仅支持将mp4的视频进行转码")
    capture = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    fps = capture.get(cv2.CAP_PROP_FPS)
    ret,frame = capture.read()
    height = frame.shape[0]
    width = frame.shape[1]
    video_name = GetLastDir(video_path)
    output_path = os.path.join(GetPreviousDir(video_path),video_name[:-4]+".avi")
    videoWriter = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    processBar = ProcessBar()
    processBar.count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    while ret:
        processBar.start_time = time.time()
        videoWriter.write(frame)
        ret,frame = capture.read()
        NoLinePrint("转码中",processBar)

if __name__ == '__main__':
    videoToAvi("/home/jade/Videos/IPS_2019-08-14.14.22.14.mp4")