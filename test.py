#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : test.py
# @Author   : jade
# @Date     : 2023/3/7 11:16
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from jade import *
def test_process_bar():
    progressBar = ProgressBar(10)
    for i in range(10):
        time.sleep(1)
        progressBar.update()
if __name__ == '__main__':
    test_process_bar()