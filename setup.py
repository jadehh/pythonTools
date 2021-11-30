#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : setup.py.py
# @Author   : opencv_tools
# @Date     : 2021/4/30 14:05
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from setuptools import setup, find_packages
if __name__ == '__main__':
    pack_list = ["opencv_tools"]
    find_packages("opencv_tools", pack_list)
    setup(
    name="opencv_tools",
    version="1.0.0",
    keywords=("pip", "opencv_tools", ""),
    description="opencv_tools",
    long_description="",
    license="MIT Licence",
    url="https://jadehh@live.com",
    author="opencv_tools",
    author_email="jadehh@live.com",

    packages=pack_list,
    package_data={'': ['*.ttf']},

    include_package_data=True,
    platforms="any",
    install_requires=["easycython==1.0.7","pyinstaller==4.6","numpy","pillow","opencv-python"]  # 这个项目需要的第三方库
)

