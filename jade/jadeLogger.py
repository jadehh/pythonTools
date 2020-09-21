#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : jadeLogger.py
# @Author   : jade
# @Date     : 2020/9/21 上午9:34
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : test.py
# @Author   : jade
# @Date     : 2020/9/21 上午9:21
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from mlogging import RotatingFileHandler_MP
from jade import CreateSavePath,get_python_version
import logging
import os


class JadeLogging:
    def __init__(self,logger_path):
        self.logger = logging.getLogger(__name__)
        # Add the log message handler to the logger
        handler = RotatingFileHandler_MP(os.path.join(logger_path,"info.log"),
                                          maxBytes=200, backupCount=5)

        self.logger.setLevel("DEBUG")
        self.logger.addHandler(handler)


    def ERROR(self,content):
        if "python2.7" == get_python_version():
            content = content.decode("utf-8")
        self.logger.error(content)

    def INFO(self,content):
        if "python2.7" == get_python_version():
            content = content.decode("utf-8")
        self.logger.info(content)

    def DEBUG(self,content):
        if "python2.7" == get_python_version():
            content = content.decode("utf-8")
        self.logger.debug(content)


if __name__ == '__main__':
    import time
    jadeLog = JadeLogging("log")
    time.sleep(2)
    jadeLog.INFO("123")
    time.sleep(2)
    jadeLog.ERROR("ERROR")
    time.sleep(2)
    jadeLog.DEBUG("结束")
