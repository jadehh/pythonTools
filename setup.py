#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : setup.py.py
# @Author   : jade
# @Date     : 2021/4/30 14:05
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
import shutil
def write_version(package_name):
    with open("{}/version.py".format(package_name),"wb") as f:
        f.write("full_version  = '{}'\n".format(get_app_version()).encode("utf-8"))
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
package_name = "jade"
write_version(package_name=package_name)
from setuptools import setup, find_packages
from jade import get_app_version,write_version,CreateSavePath
if __name__ == '__main__':
    pack_list = [package_name]
    CreateSavePath("Ouput")
    find_packages(package_name, pack_list)
    setup(
    name=package_name,
    version=get_app_version(),
    keywords=("pip", package_name, ""),
    description=package_name,
    long_description="",
    license="MIT Licence",
    url="https://jadehh@live.com",
    author="jade",
    author_email="jadehh@live.com",

    packages=pack_list,
    package_data={'': ['*Run','*png']},
    include_package_data=True,
    platforms="any",
    install_requires=["easycython","pyinstaller==4.6","cryptography==3.4.8","cffi"]  # 这个项目需要的第三方库
)



