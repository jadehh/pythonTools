#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : jadeLog.py
# @Author  : jade
# @Date    : 2019/12/19 13:56
# @Mailbox : jadehh@live.com
# @Software: Samples
# @Desc    :
import logging
import logging.config
import sys
import os
DEBUG = "debug"
ERROR = "error"
INFO = "info"
WARNING = "warning"
CRITICAL = "critical"
class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,level='info',fmt='%(asctime)s - %(levelname)s: %(message)s'):
        if not os.path.exists(os.path.abspath(os.getcwd() + "/log")):
            os.makedirs(os.path.abspath(os.getcwd() + "/log"))
        config_path = ""
        for path in sys.path:
            if 'site-packages' in path:
                config_path = path+"/jade/"+"logger_config.ini"
                break
        logging.config.fileConfig(config_path)
        self.logger = logging.getLogger(name="root")
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        self.logger.addHandler(sh) #把对象加到logger里

def JadeLog(log,content,Type="DEBUG"):
    if Type == "debug":
        log.logger.debug(content)
    elif Type == "info":
        log.logger.info(content)
    elif Type == "warning":
        log.logger.warning(content)
    elif Type == "error":
        log.logger.error(content)
        #Logger(os.path.join(log.path,"Error.log"), level="error").logger.error(content)
    elif Type == 'critical':
        log.logger.critical(content)

if __name__ == '__main__':
    log = Logger(level='debug')
    JadeLog(log,"123","error")

