#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : jade_logging.py
# @Author  : jade
# @Date    : 20-9-14 上午9:36
# @Mailbox : jadehh@live.com
# @Software: Samples
# @Desc    :
from jade.jade_tools import CreateSavePath
import time
import logging.config
from queue import Queue
from threading import Thread
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
logging.getLogger("spyne").setLevel(logging.ERROR)
logging.getLogger("spyne.application").setLevel(logging.ERROR)
logging.getLogger("spyne.application.client").setLevel(logging.ERROR)
logging.getLogger("spyne.application.server").setLevel(logging.ERROR)
logging.getLogger("suds").setLevel(logging.ERROR)
logging.getLogger("tornado.access").disabled = True
logging.getLogger("tornado.application").disabled = True
logging.getLogger("tornado.general").disabled = True


class JadeLogging():
    """
    TimedRotatingFileHandler 测试
    """
    def __init__(self,logging_path="log",max_count=180,Level="INFO"):
        CreateSavePath(logging_path)
        import os
        log_conf = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'default': {
                    'format': '%(asctime)s - %(levelname)s: %(message)s',
                    'datefmt': "%Y-%m-%d %H:%M:%S"
                },
            },
            'handlers': {
                'file': {
                    'level': Level,
                    'class': 'logging.handlers.TimedRotatingFileHandler',
                    'when': 'd',
                    'backupCount': max_count,
                    'filename': os.path.join(logging_path,"info.log"),
                    'encoding': 'utf-8',
                    'formatter': 'default',
                }
            },
            'root': {
                'handlers': ['file'],
                'level': Level,
            },
        }


        file_path = os.path.split(log_conf.get("handlers").get("file").get("filename"))[0]
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        fmt = log_conf.get("formatters").get("default").get("format")
        format_str = logging.Formatter(fmt)  # 设置日志格式
        sh = logging.StreamHandler()  # 往屏幕上输出
        sh.setFormatter(format_str)  # 设置屏幕上显示的格式
        logging.config.dictConfig(log_conf)
        self.logger = logging.getLogger(__name__)
        self.max_format = 100
        self.logger.addHandler(sh)  # 把对象加到logger里
        self.logContent = Queue(maxsize=200)
        getlogContentThread = GetLogContentThread(self.write_log,self.logContent)
        getlogContentThread.start()

    def format(self,content):
        if len(content) < self.max_format:
            if (self.max_format-len(content )) % 2 == 0:
                content = "#"*int((self.max_format-len(content )) / 2) + content + "#"*int((self.max_format-len(content )) / 2)
            else:
                content = "#" * int(self.max_format - len(content) / 2) + content + "#" * int(
                    (self.max_format - len(content)) / 2)+"#"
        return content


    def write_log(self,content, Type="debug"):
        if Type == "debug":
            self.logger.debug(content)
        elif Type == "info":
            self.logger.info(content)
        elif Type == "warning":
            self.logger.warning(content)
        elif Type == "error":
            self.logger.error(content)
        elif Type == 'critical':
            self.logger.critical(content)
    def WARNING(self,content,is_format=False):
        if is_format:
            content = self.format(content)
        self.logContent.put((content, "warning"))
    def DEBUG(self,content,is_format=False):
        if is_format:
            content = self.format(content)
        self.logContent.put((content,"debug"))
    def ERROR(self,content,is_format=False):
        if is_format:
            content = self.format(content)
        self.logContent.put((content,"error"))
    def INFO(self, content,is_format=False):
        if is_format:
            content = self.format(content)
        self.logContent.put((content,"info"))

    def release(self):
        self.logContent.put((False,"stop"))


class GetLogContentThread(Thread):
    def __init__(self,jadeLog,logcontentQueue):
        self.func = jadeLog
        self.logcontentQueue = logcontentQueue
        Thread.__init__(self)
    def run(self):
        while True:
            content,log_type = self.logcontentQueue.get()
            if content is False:
                break
            self.func(content,log_type)


if __name__ == "__main__":
    print("50W日志写入测试")
    begin_time = time.time()
    # 多进程写入日志，进程数与CPU核心数一致，使用文件锁实现进程并发控制，防止脏数据以及日志丢失
    # 每个进程100个线程共需写入五千行日志，由于GIL原因，并发只存在一个线程，但是会存在线程上下文切换，使用线程锁防止脏数据和日志丢失
    # ConcurrentTimedRotatingFileHandlerTest().mutil_process_write_log()
    # use_time = time.time() - begin_time
    # print("ConcurrentTimedRotatingFileHandler 耗时:%s秒" % use_time)
    # begin_time = time.time()
    # 每个进程100个线程共需写入所有日志，由于GIL原因，并发只存在一个线程，但是会存在线程上下文切换，同样需要锁机制防止脏数据和日志丢失
    jadeLog = JadeLogging()

    jadeLog.INFO("123")
    time.sleep(2)
    jadeLog.ERROR("ERROR")
    time.sleep(2)
    jadeLog.DEBUG("结束")


    use_time = time.time() - begin_time
    print("TimedRotatingFileHandler 耗时:%s秒" % use_time)
