#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : setup.py.py
# @Author   : dataset_tools
# @Date     : 2021/4/30 14:05
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from setuptools import setup, find_packages
if __name__ == '__main__':
    pack_list = ["dataset_tools"]
    find_packages("dataset_tools", pack_list)

    setup(
    name="dataset_tools",
    version="1.0.7",
    keywords=("pip", "dataset_tools", ""),
    description="dataset_tools",
    long_description="",
    license="MIT Licence",
    url="https://jadehh@live.com",
    author="dataset_tools",
    author_email="jadehh@live.com",

    packages=pack_list,
    package_data={'': ['*.Run']},
    include_package_data=True,
    platforms="any",
    install_requires=["easycython==1.0.7","pyinstaller==4.6","pycocotools"]  # 这个项目需要的第三方库
)

