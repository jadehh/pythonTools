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
        site_packages_path = ""
        for path in sys.path:
            if 'site-packages' in path:
                site_packages_path = path
                break
        config_path = site_packages_path.split("site-packages")[0] + "site-packages/jade/logger_config.ini"
        print(config_path)
        logging.config.fileConfig(config_path)
        self.logger = logging.getLogger(name="root")

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

