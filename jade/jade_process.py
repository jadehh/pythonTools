#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : python_pynotify_txt.py
# @Author   : jade
# @Date     : 2021/7/14 15:56
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
#coding = utf-8
import os
import pyinotify
from threading import Thread
import datetime
import json
import time
from jade import CreateSavePath,GetTimeStamp
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

    def select_no_repeat_content(self,log_text,info_path):
        if os.path.exists(info_path):
            with open(info_path, "r") as f1:
                if  log_text not in f1.read().split("\n"):
                    with open(info_path, "a") as f1:
                        f1.write(log_text)


        else:
            with open(info_path,"a") as f1:
                f1.write(log_text)



    def process_IN_MODIFY(self, event):
        new_lines = self.file.read()
        try:
            for log_str in (new_lines.split("\n")[:-1]):
                log_dict = json.loads(log_str)
                log_text = log_dict["log"]
                log_time_stamp = log_dict["time"]
                log_time_int = self.str_to_timestmp(log_time_stamp, hour=8)
                this_time_int = self.str_to_timestmp(GetTimeStamp())
                if this_time_int == log_time_int:
                    with open(os.path.join(self.save_path, "info.log"), "a") as f:
                        if self.last_time_int:
                            if self.last_time_int.day == log_time_int.day:
                                f.write(log_text)
                            else:
                                with open(os.path.join(self.save_path, "info.log"), "r") as f2:
                                    with open(os.path.join(self.save_path,
                                                           "info-{}-{}-{}.log".format(self.last_time_int.year,
                                                                                      self.last_time_int.month,
                                                                                      self.last_time_int.day)),
                                              "w") as f3:
                                        f3.write(f2.read())
                                # 复制info.log为info-year-month-day.log
                                f.truncate(0)
                                f.write(log_text)
                        else:
                            #需要判断是否重复
                            f.write(log_text)

                    self.last_time_int = log_time_int
                elif log_time_int < this_time_int:
                    #需要判断是否重复,如果在
                    ##
                    self.select_no_repeat_content(log_text,os.path.join(save_log_path,"info.log"))




        except Exception as e:
            print(new_lines,e)




class DockerLogsThread(Thread):
    def __init__(self,key_str,save_path):
        self.key_str = key_str
        self.docker_root_path = "/var/lib/docker/containers"
        self.save_path = CreateSavePath(save_path)
        self.txt_path = self.getContainerLogPath()
        self.txt_path = "data/docker_logs/017ead630441943462d3b9f1dec4f820fe479f345d81f9900ddd25f4e8b05d05-json.log"
        if self.txt_path:
            wm = pyinotify.WatchManager()
            mask = pyinotify.IN_CREATE | pyinotify.IN_MODIFY  # 还有删除等，可以查看下官网资料
            self.notifier = pyinotify.Notifier(wm, OnWriteHandler(self.txt_path, save_path))
            wm.add_watch(self.txt_path, mask, rec=True, auto_add=True)
        super(DockerLogsThread, self).__init__()


    def getContainerLogPath(self):
        file_list = os.listdir(self.docker_root_path)
        for file_name in file_list:
            for docker_file_name in os.listdir(os.path.join(self.docker_root_path,file_name)):
                if file_name+"-json.log" == docker_file_name:
                    with open(os.path.join(self.docker_root_path,file_name,docker_file_name),"r") as f:
                        content_list = f.read().split("\n")
                        for content in content_list:
                            if  self.key_str  in content:
                                f.close()
                                return os.path.join(self.docker_root_path,file_name,docker_file_name)


    def run(self):
        while True:
            try:
                if self.txt_path is None:
                    break
                self.notifier.process_events()
                if  self.notifier.check_events():
                    self.notifier.read_events()
            except KeyboardInterrupt:
                continue




class WriteTxt(Thread):
    def __init__(self,path):
        self.index = 0
        self.path = path
        super(WriteTxt, self).__init__()

    def getTime(self,hour):
        monidatatime = datetime.datetime.now() + datetime.timedelta(hours=hour-8)
        return monidatatime.strftime('%Y-%m-%dT%H:%M:%S')
    def run(self):
        while True:
            with open(self.path,"a") as f:
                json_dict = json.dumps({"log":"{}test\n".format(self.getTime(self.index+8)),"stream":"stderr","time":self.getTime(self.index)})
                f.write(json_dict+"\n")
                print("write",json_dict)
            self.index = self.index + 1
            time.sleep(10)






if __name__ == '__main__':
    save_log_path = "/home/jade/sda2/"
    dockerLogsThread = DockerLogsThread("箱号服务已经开启,端口号为900",save_log_path)
    dockerLogsThread.start()

    writeTxt = WriteTxt("data/docker_logs/017ead630441943462d3b9f1dec4f820fe479f345d81f9900ddd25f4e8b05d05-json.log")
    writeTxt.start()