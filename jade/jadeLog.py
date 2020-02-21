#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : jadeLog.py
# @Author  : jade
# @Date    : 2019/12/19 13:56
# @Mailbox : jadehh@live.com
# @Software: Samples
# @Desc    :
import logging
from logging import handlers
from jade import GetPreviousDir
import os
class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        self.path = GetPreviousDir(filename)
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)#设置文件里写入的格式
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)

def JadeLog(log,content,Type="DEBUG"):
    if Type == "debug":
        log.logger.debug(content)
    elif Type == "info":
        log.logger.info(content)
    elif Type == "warning":
        log.logger.warning(content)
    elif Type == "error":
        log.logger.error(content)
        Logger(os.path.join(log.path,"Error.log"), level="error").logger.error("error")
    elif Type == 'critical':
        log.logger.critical(content)

if __name__ == '__main__':
    log = Logger('/home/jade/PycharmProjects/Gitee/pythonTools/jade/all.log',level='debug')
    Logger('error.log', level='error').logger.error('error')
    JadeLog(log,"123","error")

