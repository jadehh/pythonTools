#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : jade_process.py
# @Author   : jade
# @Date     : 2021/7/14 16:43
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
import os
import pyinotify
from threading import Thread
import datetime
import json
import time
import subprocess
from jade import CreateSavePath
class OnWriteHandler(pyinotify.ProcessEvent):
    def __init__(self,txt_path,save_path):
        self.save_path = save_path
        self.txt_path = txt_path
        self.last_time_int = None
        self.file = open(self.txt_path)
    def process_IN_CREATE(self, event):
        print ("create file: %s " % os.path.join(event.path,event.name))

    def str_to_timestmp(self, str,hour=None):
        str = str[0:19]
        if "T" in str:
            stripdatetime = datetime.datetime.strptime(str, '%Y-%m-%dT%H:%M:%S')

        else:
            stripdatetime = datetime.datetime.strptime(str, "%Y-%m-%d %H:%M:%S")

        if hour:
            delta_hour =  datetime.timedelta(hours=hour)
            timeStamp = stripdatetime + delta_hour
            return timeStamp
        timeStamp = stripdatetime
        return timeStamp


    def process_IN_MODIFY(self, event):
        new_lines = self.file.read()
        try:
            for log_str in (new_lines.split("\n")[:-1]):
                log_dict = json.loads(log_str)
                log_text = log_dict["log"]
                log_time_stamp = log_dict["time"]
                log_time_int = self.str_to_timestmp(log_time_stamp, hour=8)
                ##info.log文件只能是当天的日志,每次写入的时候,都需要判断上一次写入的内容
                with open(os.path.join(self.save_path,"info.log"), "a") as f:
                    if self.last_time_int:
                        if self.last_time_int.day == log_time_int.day:
                            f.write(log_text)
                        else:
                            with open(os.path.join(self.save_path,"info.log"), "r") as f2:
                                with open(os.path.join(self.save_path, "info-{}-{}-{}.log".format(self.last_time_int.year, self.last_time_int.month,
                                                                     self.last_time_int.day)), "w") as f3:
                                    f3.write(f2.read())
                            # 复制info.log为info-year-month-day.log
                            f.truncate(0)
                            f.write(log_text)
                    else:
                        f.write(log_text)

                self.last_time_int = log_time_int
        except Exception as e:
            print(new_lines,e)



class DockerLogsThread(Thread):
    def __init__(self,container_name,save_path):
        self.docker_root_path = "/var/lib/docker/containers"
        self.save_path = CreateSavePath(save_path)
        self.container_name = container_name
        self.container_id = self.getContainerID()
        self.container_path = self.getContainerPath()
        self.txt_path = self.getLogPath()
        if self.txt_path:
            if os.path.exists(os.path.join(self.save_path, "info.log")):
                os.remove(os.path.join(self.save_path, "info.log"))
            wm = pyinotify.WatchManager()
            mask = pyinotify.IN_CREATE | pyinotify.IN_MODIFY  # 还有删除等，可以查看下官网资料
            self.notifier = pyinotify.Notifier(wm, OnWriteHandler(self.txt_path, save_path))
            wm.add_watch(self.txt_path, mask, rec=True, auto_add=True)

        super(DockerLogsThread, self).__init__()

    def getLogPath(self):
        for file_name in os.listdir(self.container_path):
            if self.container_id in file_name:
                log_path = os.path.join(self.container_path,file_name)
                return log_path
    def getContainerID(self):
        cmd_str = "docker ps -aqf 'name={}'".format(self.container_name)
        result_bytes = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return str(result_bytes.stdout.readlines()[0], encoding="utf-8").split("\n")[0]

    def getContainerPath(self):
        try:
            for filename in os.listdir(self.docker_root_path):
                if self.container_id in filename:
                    return os.path.join(self.docker_root_path, filename)
        except Exception as e:
            print(e)
    def run(self):
        while True:
            if self.txt_path is None:
                break
            try:
                self.notifier.process_events()
                if  self.notifier.check_events():
                    self.notifier.read_events()
            except KeyboardInterrupt:
                continue


if __name__ == '__main__':
    container_name = "container_ocrV2.2-{}".format(1)
    save_log_path = "/home/jade/sda2/LOG" + "_test"
    dockerLogsThread = DockerLogsThread(container_name,save_log_path)
    dockerLogsThread.start()