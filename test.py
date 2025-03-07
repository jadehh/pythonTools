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

# 如果保存路径不存在，则创建
if os.path.exists(save_path) is False:
    os.makedirs(save_path)


def test_download_file_with_ie_time(ie_time, front_container_number, rear_container_number):
    # 下载录像
    try:
        start_time_dt = datetime.strptime(ie_time, "%Y-%m-%d %H:%M:%S.%f")
        start_time_str = (start_time_dt - timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M:%S")
        end_time_str = (start_time_dt + timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M:%S")
        save_name = f"{ie_time.replace('-', '').replace(':', '').replace(' ', '').replace('.', '')}_{front_container_number}_{rear_container_number}.mp4"
        nvr_client.download_recording(os.path.join(save_path, save_name), start_time_str, end_time_str)  # 下载录像
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
            if container_count > 0:
                test_download_file_with_ie_time(ie_time, front_container_number, rear_container_number)


if __name__ == "__main__":
    test_download_file_from_file("过卡记录.txt")
