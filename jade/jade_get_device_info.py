#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : GetDeviceINFOThread.py
# @Author  : jade
# @Date    : 2021/1/17 下午6:17
# @Mailbox : jadehh@live.com
# @Software: Samples
# @Desc    :
import pynvml
import psutil
from threading import Thread
import sys
import time

class GetDeviceINFOThread(Thread):
    def __init__(self,pid):
        self.pid = pid
        self.StartStatus = True
        self.index = 0
        self.sum_cpu_per = 0
        self.sum_mem_per = 0
        self.sum_io_cnt = 0
        self.sum_gpu_per = 0
        self.sum_gpu_mem = 0
        self.mean_cpu_per = 0
        self.mean_mem_per = 0
        self.mean_io_cnt = 0
        self.mean_gpu_per = 0
        self.mean_gpu_mem = 0
        self.NUM_EXPAND = 1024 * 1024

        pynvml.nvmlInit()
        self.pro = psutil.Process(self.pid)
        Thread.__init__(self)
        self.start()

    def release(self):
        self.index = 0
        self.sum_cpu_per = 0
        self.sum_mem_per = 0
        self.sum_io_cnt = 0
        self.sum_gpu_per = 0
        self.sum_gpu_mem = 0
        self.mean_cpu_per = 0
        self.mean_mem_per = 0
        self.mean_io_cnt = 0
        self.mean_gpu_per = 0
        self.mean_gpu_mem = 0

    def getInfo(self):
        self.mean_cpu_per = self.sum_cpu_per / self.index
        self.mean_gpu_per = self.sum_gpu_per / self.index
        self.mean_mem_per = self.sum_mem_per  / self.index
        self.mean_io_cnt = self.sum_io_cnt  / self.index
        self.mean_gpu_mem = self.sum_gpu_mem / self.index
        return round(self.mean_cpu_per,2),round(self.mean_gpu_per, 2),round(self.mean_mem_per, 2),round(self.mean_io_cnt, 2), round(self.mean_gpu_mem, 2)
    def stop(self):
        time.sleep(1)
        self.StartStatus = False
    def run(self):
        while self.StartStatus:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info_mem_list = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)  # 获取所有GPU上正在运行的进程信息
            info_list_len = len(info_mem_list)
            gpu_mem= 0
            if info_list_len > 0:  # 0表示没有正在运行的进程
                for info_i in info_mem_list:
                    if info_i.pid == self.pid:  # 如果与需要记录的pid一致
                        gpu_mem += info_i.usedGpuMemory / self.NUM_EXPAND  # 统计某pid使用的总显存
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_percent = util.gpu
            cpu_per = self.pro.cpu_percent()
            mem = self.pro.memory_percent()
            io = self.pro.io_counters()
            io_cnt = io.read_count + io.write_count
            self.sum_cpu_per  = self.sum_cpu_per + cpu_per
            self.sum_mem_per = self.sum_mem_per + mem
            self.sum_io_cnt = self.sum_io_cnt + io_cnt
            self.sum_gpu_mem = self.sum_gpu_mem + gpu_mem
            self.sum_gpu_per = self.sum_gpu_per + gpu_percent
            self.index = self.index + 1


if __name__ == '__main__':
    import os
    getDeviceInfoThread = GetDeviceINFOThread(os.getpid())
    getDeviceInfoThread.start()