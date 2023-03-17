#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : setup.py
# @Author   : jade
# @Date     : 2023/3/17 9:55
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from setuptools import setup, find_packages
from jade import getOperationSystem,CreateSavePath
import platform
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
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str,
                        default=[])  ## 打包名称
    parser.add_argument('--path', type=str,
                        default="")

    args = parser.parse_args()
    package_name = "pyldk"
    write_version(package_name)
    ## 打包区分操作系统和arch
    ##一共有4个版本 linux两个版本,windows 两个版本

    version_list = [{"name":"{}-{}-py3-none-manylinux1_x86_64.whl","path":"pyldk/lib/Linux/x86_64"},
                    {"name":"{}-{}-py3-none-manylinux2014_aarch64.whl","path":"pyldk/lib/Linux/aarch64"},
                    {"name":"{}-{}-py3-none-win32.whl","path":"pyldk/lib/Windows/x86"},
                    {"name":"{}-{}-py3-none-win_amd64.whl","path":"pyldk/lib/Windows/x64"}
                    ]
    CreateSavePath("Ouput")
    pack_list = [package_name,args.path]
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
        package_data={'': ['*.so']},

        include_package_data=True,
        platforms="any",
        install_requires=[]  # 这个项目需要的第三方库
    )

    shutil.copy("dist/{}-{}-py3-none-any.whl".format(package_name, get_app_version()),
                "Ouput/{}-{}-py3-none-{}.whl".format(package_name, get_app_version(),args.name))
    shutil.rmtree("dist")
    shutil.rmtree("build")
    shutil.rmtree("{}.egg-info".format(package_name))


