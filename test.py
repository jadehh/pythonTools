"""
# @File     : test.py
# @Author   : jade
# @Date     : 2025/3/7 10:47
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     : test.py
"""
# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from datetime import datetime, timedelta
from src.nvr import NVRClient

# 初始化 NVR 客户端
nvr_ip = "192.168.29.110"  # 替换为你的 NVR IP 地址
username = "admin"  # 替换为你的 NVR 用户名
password = "samples456"  # 替换为你的 NVR 密码
save_path = r"DVRVideo/"  # 保存路径
channel_number_index = 0  # 通道号索引
nvr_client = NVRClient(nvr_ip, username, password, channel_number_index)
nvr_client.connect()
channels = nvr_client.get_channel_numbers()
channel_number = channels[channel_number_index]

nvr_start_time_str = "2024-12-30 00:00:00.000"  # 硬盘录像机的开始时间
nvr_end_time_str = "2025-01-15 14:16:07.000"  # 硬盘录像机的结束时间


# 如果保存路径不存在，则创建
if os.path.exists(save_path) is False:
    os.makedirs(save_path)


def test_download_file_with_ie_time(ie_time, front_container_number, rear_container_number):
    # 下载录像
    try:
        """
        每天时间都会变化，所以需要根据偏移时间来下载
        天数越多，偏移时间越大
        例如顾卡时间为:2024-12-31 08:27:14, 偏移时间为: 30, 开始时间需要往前偏移30s,结束时间往后偏移60s
        例如顾卡时间为:2024-01-01 08:27:14, 偏移时间为: 30 - 1*4, 开始时间需要往前偏移30s,结束时间往后偏移60s
        例如过卡时间为:2025-01-13 15:29:45, 偏移时间为: 30 - 4*14  开始时间需要往后偏移30s,结束时间在往后偏移60s
        """

        start_time_dt = datetime.strptime(ie_time, "%Y-%m-%d %H:%M:%S.%f")
        # 加入时间判断，时间需要在白天时间
        if start_time_dt.hour < 7 or start_time_dt.hour > 16:
            print(f"时间为:{ie_time}，时间不在白天时间内，不下载。")
        else:
            time_dt = datetime.strptime("2024-12-31", "%Y-%m-%d") # 基准时间
            all_time = 60  # 总时长
            daily_offset_time = 4 # 每天偏移时间
            offset_time = 30 # 默认偏移时间,代表基准时间偏移的时间为30
            start_offset_time = (offset_time - (start_time_dt - time_dt).days * daily_offset_time)
            end_offset_time = all_time - start_offset_time  # 偏移时间
            start_time_str = (start_time_dt - timedelta(seconds=start_offset_time)).strftime("%Y-%m-%d %H:%M:%S")
            end_time_str = (start_time_dt + timedelta(seconds=end_offset_time)).strftime("%Y-%m-%d %H:%M:%S")
            save_name = f"{ie_time.replace('-', '').replace(':', '').replace(' ', '').replace('.', '')}_{front_container_number}{'' if len(rear_container_number) == 0 else '_' + rear_container_number}.mp4"
            print( f"开始下载,到达时间为:{ie_time},偏移时间为:{start_offset_time},开始时间:{start_time_str},结束时间:{end_time_str}")
            nvr_client.download_recording(channel_number, os.path.join(save_path, save_name), start_time_str,
                                          end_time_str)  # 下载录像
            print(f"录像成功下载到 {os.path.join(save_path, save_name)}。")
    except ValueError as e:
        raise ValueError("无效的开始时间格式。请使用 'YYYY-MM-DD HH:MM:SS.fff' 格式。") from e


def test_download_file_from_file(file_path):
    # 从文件中读取数据并下载录像
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            datas = line.strip().split(',')
            ie_time = datas[39]
            front_container_number = datas[25]  # 前箱号
            rear_container_number = datas[26]  # 后箱号
            container_count = 0 if datas[23] == "NULL" else int(datas[23])
            if datetime.strptime(ie_time, "%Y-%m-%d %H:%M:%S.%f") > datetime.strptime(nvr_start_time_str,"%Y-%m-%d %H:%M:%S.%f") and datetime.strptime(ie_time, "%Y-%m-%d %H:%M:%S.%f") < datetime.strptime(nvr_end_time_str, "%Y-%m-%d %H:%M:%S.%f"):
                if container_count > 0:
                    if len(front_container_number) == 0:
                        front_container_number = "Unrecognized"
                    if container_count == 2 and len(rear_container_number) == 0:
                        rear_container_number = "Unrecognized"
                    test_download_file_with_ie_time(ie_time, front_container_number, rear_container_number)
            else:
                print(f"过卡时间为: {ie_time}, 硬盘录像机开始时间为: {nvr_start_time_str}, 硬盘录像机结束时间为: {nvr_end_time_str}, 不在硬盘录像机时间范围内。")
if __name__ == "__main__":
    test_download_file_from_file("过卡记录.txt")
    # test_download_file_with_ie_time("2025-01-04 16:18:28.000","SEGU6713630","")
