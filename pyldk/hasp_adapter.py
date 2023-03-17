#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : samplesMain.py
# @Author   : jade
# @Date     : 2023/3/15 14:58
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
import os
import shutil
from ctypes import *
from jade import *
import ctypes
class HaspStruct(Structure):
	_fields_ = [('status',c_int),('info',c_char_p)]
class BaseAdapter():
    # 动态sdk文件 .so .dll
    def __init__(self,so_path,JadeLog=None):
        self.so_path = so_path
        self.JadeLog = JadeLog
        self.lib = CDLL(self.so_path)


    def log(self, msg, log_level="ERROR"):
        if self.JadeLog:
            if log_level == "INFO":
                self.JadeLog.INFO(msg)
            elif log_level == "ERROR":
                self.JadeLog.ERROR(msg)
        else:
            print(msg)

    def get_info(self):
        feature_id = ""
        try:
            self.lib.getInfo.restype = HaspStruct
            haspStruct = self.lib.getInfo()
            if haspStruct.status == 0:
                try:
                    feature_id_list = str(haspStruct.info,encoding="utf-8").split("<feature id=")[1:]
                    for feature_id_str in feature_id_list:
                        feature_id = int((feature_id_str.split("/>\n")[0].split('"')[1]))
                        if feature_id != 0:
                            break
                except Exception as e:
                    self.log("获取加密狗ID失败,失败原因为:{}".format(e))
            else:
                self.log("获取加密狗ID失败,失败代码为:{}".format(haspStruct.status))
        except Exception as e:
            self.log("获取加密狗ID列表失败,失败原因为:{}".format(e))
        return feature_id

    def show_staus(self,operation_str,status):
        if status == 7:
            self.log("{},请检查加密狗是否正常插入".format(operation_str))
        elif status == 38:
            self.log("{},请检查登录用户是否超出授权的最大用户数".format(operation_str))
        elif status == 50:
            self.log("{},加密狗中途断开".format(operation_str))
        elif status == 84:
            self.log("{},共享违规,请重启加密狗服务".format(operation_str))

        else:
            self.log("{},失败代码为:{}".format(operation_str,status))


    def bool_lower_users(self,feature_id):
        is_lower_users = False
        self.lib.getSessionInfo.restype = HaspStruct
        try:
            haspStruct = self.lib.getSessionInfo(feature_id)
            if haspStruct.status == 0:
                try:
                    maxlogins = int(str(haspStruct.info,encoding="utf-8").split("<maxlogins>")[-1].split("</maxlogins>")[0])
                    currentlogins = int(str(haspStruct.info,encoding="utf-8").split("<currentlogins>")[-1].split("</currentlogins>")[0])
                    if currentlogins < maxlogins:
                        return True
                    else:
                        return False
                except Exception:
                    self.log("获取加密狗登录最大用户失败,失败原因为:{}".format(e))
            else:
                self.show_staus("获取加密狗登录最大用户失败",haspStruct.status)
        except Exception as e:
            self.log("获取是否超出加密狗登录最大用户失败,失败原因为:{}".format(e))


    def login(self):
        status = self.lib.login()
        return status
