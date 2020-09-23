#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : jadeLogging.py
# @Author  : jade
# @Date    : 20-9-14 上午9:36
# @Mailbox : jadehh@live.com
# @Software: Samples
# @Desc    :
import time
from concurrent.futures import  ThreadPoolExecutor
from jade import get_python_version,CreateSavePath

class JadeLogging:
    """
    TimedRotatingFileHandler 测试
    """

    def __init__(self,logging_path="log"):
        CreateSavePath(logging_path)
        import logging.config
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
                    'level': 'DEBUG',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'maxBytes': 100000,
                    'backupCount': 1000,
                    'delay': True,
                    'filename': os.path.join(logging_path,"info.log"),
                    'encoding': 'utf-8',
                    'formatter': 'default',
                }
            },
            'root': {
                'handlers': ['file'],
                'level': 'DEBUG',
            },
        }

        import os
        file_path = os.path.split(log_conf.get("handlers").get("file").get("filename"))[0]
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        fmt = log_conf.get("formatters").get("default").get("format")
        format_str = logging.Formatter(fmt)  # 设置日志格式
        sh = logging.StreamHandler()  # 往屏幕上输出
        sh.setFormatter(format_str)  # 设置屏幕上显示的格式
        logging.config.dictConfig(log_conf)
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(sh)  # 把对象加到logger里


    def write_log(self,content, Type="debug"):
        if "python2.7" == get_python_version():
            content = content.decode("utf-8")
        if Type == "debug":
            self.logger.debug(content)
        elif Type == "info":
            self.logger.info(content)
        elif Type == "warning":
            self.logger.warning(content)
        elif Type == "error":
            self.logger.error(content)
            # Logger(os.path.join(log.path,"Error.log"), level="error").logger.error(content)
        elif Type == 'critical':
            self.logger.critical(content)

    def mutil_thread_write_log(self):
        with ThreadPoolExecutor(max_workers=100) as thread_pool:
            for i in range(100):
                thread_pool.submit(self.write_log, i,"debug").add_done_callback(self._executor_callback)

    def DEBUG(self,content):
        with ThreadPoolExecutor(max_workers=100) as thread_pool:
            thread_pool.submit(self.write_log, content,"debug").add_done_callback(self._executor_callback)

    def ERROR(self,content):
        with ThreadPoolExecutor(max_workers=100) as thread_pool:
            thread_pool.submit(self.write_log, content, "error").add_done_callback(self._executor_callback)

    def INFO(self, content):
        with ThreadPoolExecutor(max_workers=100) as thread_pool:
            thread_pool.submit(self.write_log, content, "info").add_done_callback(self._executor_callback)

    def _executor_callback(self, worker):
        worker_exception = worker.exception()
        if worker_exception:
            print("Worker return exception: ", self.worker_exception)




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
    jadeLog.mutil_thread_write_log()
    time.sleep(2)
    jadeLog.INFO("123")
    jadeLog.ERROR("ERROR")
    time.sleep(2)
    jadeLog.DEBUG("结束")
    use_time = time.time() - begin_time
    print("TimedRotatingFileHandler 耗时:%s秒" % use_time)
