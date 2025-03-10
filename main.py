"""
# @File     : main.py
# @Author   : jade
# @Date     : 2025/3/7 10:47
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     : main.py
"""
# !/usr/bin/env python
# -*- coding: utf-8 -*-
from src.nvr import NVRClient


def main():
    # 初始化 NVR 客户端
    nvr_ip = "192.168.29.110"  # 替换为你的 NVR IP 地址
    username = "admin"  # 替换为你的 NVR 用户名
    password = "samples456"  # 替换为你的 NVR 密码
    # 下载录像
    start_time_str = "2024-12-31 08:00:00"  # 开始时间
    end_time_str = "2024-12-31 08:00:20"  # 结束时间
    save_path = "recording.mp4"  # 替换为所需的保存路径
    channel_number_index = 0  # 通道号索引
    nvr_client = NVRClient(nvr_ip, username, password, channel_number_index)
    nvr_client.download_recording(nvr_client.get_channel_numbers()[channel_number_index],save_path, start_time_str, end_time_str)  # 下载录像
    print(f"录像成功下载到 {save_path}。")  # 打印下载成功信息


if __name__ == "__main__":
    main()
