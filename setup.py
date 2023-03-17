#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : setup.py
# @Author   : jade
# @Date     : 2023/3/17 9:55
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from setuptools import setup, find_packages
def get_app_version():
    try:
        with open("CONTRIBUTING.md","rb") as f:
            content = str(f.read(),encoding="utf-8").split("#### ")[1].split(" - ")[0]
            version = ""
            if "v" in content and "V" in content:
                version = content.split("V")[-1]
            elif "v" in content:
                version = content.split("v")[-1]
            elif "V" in content:
                version = content.split("V")[-1]
            if version:
                return version
            else:
                raise "please check CONTRIBUTING contain version"
    except:
        raise "please check CONTRIBUTING contain version"
if __name__ == '__main__':
    pack_list = ["pyldk"]
    find_packages("pyldk", pack_list)
    setup(
    name="pyldk",
    version=get_app_version(),
    keywords=("pip", "pyldk", ""),
    description="pyldk",
    long_description="",
    license="MIT Licence",
    url="https://jadehh@live.com",
    author="jade",
    author_email="jadehh@live.com",

    packages=pack_list,
    package_data={'': ['*.so']},

    include_package_data=True,
    platforms="any",
    install_requires=[]  # 这个项目需要的第三方库
)

