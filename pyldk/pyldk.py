#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : pyldk.py
# @Author   : jade
# @Date     : 2023/3/16 16:29
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from pyldk.hasp_adapter import *
import platform
class PyLdk(object):
    def __init__(self,JadeLog=None):
        self.JadeLog = JadeLog
        if getOperationSystem() == "Windows":
            if (platform.architecture()[0]) == "64bit":
                libs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'lib/{}/x64'.format(getOperationSystem())))
            else:
                libs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'lib/{}/x86'.format(getOperationSystem())))

        else:
            libs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'lib/{}/{}'.format(getOperationSystem(),self.get_system_arch())))
        self.adapter = BaseAdapter(os.path.join(libs_dir,os.listdir(libs_dir)[0]),JadeLog)

    def get_system_arch(self):
        return (platform.uname().processor)

    def log(self, msg, log_level="ERROR"):
        if self.JadeLog:
            if log_level == "INFO":
                self.JadeLog.INFO(msg)
            elif log_level == "ERROR":
                self.JadeLog.ERROR(msg)
        else:
            print(msg)

    def get_feature_id(self):
        feature_id = self.adapter.get_info()
        return feature_id

    def login(self,feature_id=None):
        ## 首先判断加密狗是否存在
        login_status = False
        if feature_id is None:
            feature_id = 0
        haspStruct = self.adapter.login(feature_id)
        if haspStruct.status == 0:
            self.adapter.logout(haspStruct.handle)
            if feature_id == 0:
                feature_id = self.get_feature_id()
            haspStruct = self.adapter.login(feature_id)
            if haspStruct.status == 0:
                if feature_id > 0:
                    login_status = True
                else:
                    self.adapter.log("加密狗初始化失败,请检查加密狗的授权,或者重新授权")
                    self.adapter.logout(haspStruct.handle)
            else:
                self.adapter.log("加密狗初始化失败",haspStruct.status)
        else:
            self.adapter.show_staus("加密狗初始化失败", haspStruct.status,feature_id)
        return haspStruct,feature_id,login_status


    def get_ldk(self,feature_id):
        ldk_status = False
        lower_users = self.adapter.bool_lower_users(feature_id)
        if lower_users:
            ldk_status = True
        return ldk_status