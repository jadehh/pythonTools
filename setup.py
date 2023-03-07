#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : setup.py.py
# @Author   : jade
# @Date     : 2021/4/30 14:05
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from setuptools import setup, find_packages
if __name__ == '__main__':
    pack_list = ["jade"]
    find_packages("jade", pack_list)
    setup(
    name="jade",
    version="1.6.3",
    keywords=("pip", "jade", ""),
    description="jade",
    long_description="",
    license="MIT Licence",
    url="https://jadehh@live.com",
    author="jade",
    author_email="jadehh@live.com",

    packages=pack_list,
    package_data={'': ['*Run','*png']},
    include_package_data=True,
    platforms="any",
    install_requires=["easycython","pyinstaller","cryptography==3.4.8","cffi"]  # 这个项目需要的第三方库
)

