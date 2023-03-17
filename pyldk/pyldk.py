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

    def get_ldk(self):
        ldk_status = False
        status = self.adapter.login()
        if status == 0:
            feature_id = self.adapter.get_info()
            if feature_id > 0:
                lower_users = self.adapter.bool_lower_users(feature_id)
                if lower_users:
                    ldk_status = True
            elif feature_id == 0:
                self.log("加密狗没有授权,请重新授权")
        else:
            self.adapter.show_staus("登录失败",status)
        return ldk_status