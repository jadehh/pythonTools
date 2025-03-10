"""
# @File     : nvr.py
# @Author   : jade
# @Date     : 2025/3/7 10:47
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     : nvr.py
"""
# !/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from ctypes import *
import os
from datetime import datetime

# 加载 HCNetSDK 库
HCNetSDKPath = os.path.join(os.getcwd(), "lib", "Windows")
HCNetSDK = cdll.LoadLibrary(os.path.join(HCNetSDKPath, 'HCNetSDK.dll'))

os.environ['PATH'] = HCNetSDKPath + ';' + os.environ['PATH']

# 定义常量
NET_DVR_PLAYSTART = 1
NET_DVR_PLAYSTOP = 2
NET_DVR_PLAYGETPOS = 3
SERIALNO_LEN = 48
STREAM_ID_LEN = 32


# 定义设备信息结构体
class NET_DVR_DEVICEINFO_V30(Structure):
    _fields_ = [
        ("sSerialNumber", c_byte * SERIALNO_LEN),  # 序列号
        ("byAlarmInPortNum", c_byte),  # 报警输入个数
        ("byAlarmOutPortNum", c_byte),  # 报警输出个数
        ("byDiskNum", c_byte),  # 硬盘个数
        ("byDVRType", c_byte),  # 设备类型
        ("byChanNum", c_byte),  # 模拟通道个数
        ("byStartChan", c_byte),  # 起始通道号
        ("byAudioChanNum", c_byte),  # 语音通道数
        ("byIPChanNum", c_byte),  # 最大数字通道个数，低位
        ("byZeroChanNum", c_byte),  # 零通道编码个数
        ("byMainProto", c_byte),  # 主码流传输协议类型
        ("bySubProto", c_byte),  # 子码流传输协议类型
        ("bySupport", c_byte),  # 能力
        ("bySupport1", c_byte),  # 能力集扩充
        ("bySupport2", c_byte),  # 能力
        ("wDevType", c_ushort),  # 设备型号
        ("bySupport3", c_byte),  # 能力集扩展
        ("byMultiStreamProto", c_byte),  # 是否支持多码流
        ("byStartDChan", c_byte),  # 起始数字通道号
        ("byStartDTalkChan", c_byte),  # 起始数字对讲通道号
        ("byHighDChanNum", c_byte),  # 数字通道个数，高位
        ("bySupport4", c_byte),  # 能力集扩展
        ("byLanguageType", c_byte),  # 支持语种能力
        ("byVoiceInChanNum", c_byte),  # 音频输入通道数
        ("byStartVoiceInChanNo", c_byte),  # 音频输入起始通道号
        ("bySupport5", c_byte),  # 能力
        ("bySupport6", c_byte),  # 能力
        ("byMirrorChanNum", c_byte),  # 镜像通道个数
        ("wStartMirrorChanNo", c_ushort),  # 起始镜像通道号
        ("bySupport7", c_byte),  # 能力
        ("byRes2", c_byte)  # 保留
    ]


# 定义时间结构体
class NET_DVR_TIME(Structure):
    _fields_ = [
        ("dwYear", c_uint),    # 年
        ("dwMonth", c_uint),   # 月
        ("dwDay", c_uint),     # 日
        ("dwHour", c_uint),    # 时
        ("dwMinute", c_uint),  # 分
        ("dwSecond", c_uint)   # 秒
    ]

# 定义播放条件结构体
class NET_DVR_PLAYCOND(Structure):
    _fields_ = [
        ("dwChannel", c_uint),  # 通道号
        ("struStartTime", NET_DVR_TIME),  # 开始时间
        ("struStopTime", NET_DVR_TIME),  # 结束时间
        ("byDrawFrame", c_byte),  # 抽帧标志，0:不抽帧，1：抽帧
        ("byStreamType", c_byte),  # 码流类型，0-主码流 1-子码流 2-码流三
        ("byStreamID", c_byte * STREAM_ID_LEN),  # 码流ID
        ("byCourseFile", c_byte),  # 课程文件标志，0-否，1-是
        ("byDownload", c_byte),  # 下载标志，0-否，1-是
        ("byOptimalStreamType", c_byte),  # 最优码流类型标志，0-否，1-是
        ("byRes", c_byte * 27)  # 保留
    ]

# 定义 NVR 客户端类
class NVRClient:
    def __init__(self, ip_address, username, password, channel_number_index):
        self.ip_address = ip_address
        self.username = username
        self.password = password
        self.channel_number_index = channel_number_index
        self.user_id = -1
        self.device_info = NET_DVR_DEVICEINFO_V30()
        self.is_connect = False

    def connect(self):
        # 初始化并连接到 NVR
        HCNetSDK.NET_DVR_Init()
        HCNetSDK.NET_DVR_SetConnectTime(5000, 3)
        HCNetSDK.NET_DVR_SetReconnect(10000, True)
        self.user_id = HCNetSDK.NET_DVR_Login_V30(self.ip_address.encode('utf-8'), 8000, self.username.encode('utf-8'),
                                                  self.password.encode('utf-8'), byref(self.device_info))

        if self.user_id < 0:
            print("登录失败，错误码:", HCNetSDK.NET_DVR_GetLastError())
            return False

        serial_number = ''.join([chr(b) for b in self.device_info.sSerialNumber if b != 0])
        print("设备信息:")
        print(f"序列号: {serial_number}")
        print(f"起始通道: {self.device_info.byStartDChan}")
        print(f"通道数量: {self.device_info.byIPChanNum}")
        print(f"设备类型: {self.device_info.byDVRType}")
        print(f"磁盘数量: {self.device_info.byDiskNum}")
        print(f"报警输入端口数量: {self.device_info.byAlarmInPortNum}")
        print(f"报警输出端口数量: {self.device_info.byAlarmOutPortNum}")
        self.is_connect = True

    def download_recording(self, channel_number,save_path, start_time_str, end_time_str):
        # 下载录像
        if self.is_connect is False:
            return False

        start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")

        start_time_struct = NET_DVR_TIME()
        start_time_struct.dwYear = start_time.year
        start_time_struct.dwMonth = start_time.month
        start_time_struct.dwDay = start_time.day
        start_time_struct.dwHour = start_time.hour
        start_time_struct.dwMinute = start_time.minute
        start_time_struct.dwSecond = start_time.second

        end_time_struct = NET_DVR_TIME()
        end_time_struct.dwYear = end_time.year
        end_time_struct.dwMonth = end_time.month
        end_time_struct.dwDay = end_time.day
        end_time_struct.dwHour = end_time.hour
        end_time_struct.dwMinute = end_time.minute
        end_time_struct.dwSecond = end_time.second

        play_cond = NET_DVR_PLAYCOND()
        play_cond.dwChannel = channel_number
        play_cond.struStartTime = start_time_struct
        play_cond.struStopTime = end_time_struct

        download_handle = HCNetSDK.NET_DVR_GetFileByTime_V40(self.user_id, save_path.encode('utf-8'), byref(play_cond))
        if download_handle < 0:
            print(f"下载失败，错误码: {HCNetSDK.NET_DVR_GetLastError()}")
            return False

        if not HCNetSDK.NET_DVR_PlayBackControl(download_handle, NET_DVR_PLAYSTART, 0, None):
            print(f"开始下载失败，错误码: {HCNetSDK.NET_DVR_GetLastError()}")
            return False

        start = time.perf_counter()
        nprog = 0
        while True:
            if nprog >= 100:
                break
            nprog = HCNetSDK.NET_DVR_GetDownloadPos(download_handle)
            finsh = "▓" * nprog
            need_do = "-" * (100 - nprog)
            progress = (nprog / 100) * 100
            dur = time.perf_counter() - start
            print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(progress, finsh, need_do, dur), end="")
            time.sleep(0.05)
            if progress < 0:
                print(f"下载过程中出错，错误码: {HCNetSDK.NET_DVR_GetLastError()}")
                break
        print("下载完成")
        return True

    def get_channel_numbers(self):
        # 获取通道号
        start_channel = self.device_info.byStartDChan
        channel_count = self.device_info.byIPChanNum
        print(f"起始通道: {start_channel}")
        print(f"通道数量: {channel_count}")
        return [start_channel + i for i in range(channel_count)]

    def disconnect(self):
        # 断开连接并清理
        if self.user_id >= 0:
            HCNetSDK.NET_DVR_Logout(self.user_id)
        HCNetSDK.NET_DVR_Cleanup()
