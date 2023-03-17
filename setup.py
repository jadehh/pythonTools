#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : write_setup.py.py
# @Author   : jade
# @Date     : 2023/3/17 13:55
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
def write_setup(args):
    with open("setup_test.py","wb") as f:
        f.write(('from setuptools import setup, find_packages\n'
                'from jade import *\n'
                'if __name__ == "__main__":\n'
                '\tpackage_name = "{}" \n'.format(args.package_name) +
                '\tpack_list = [package_name,"{}"]\n'.format(args.path) +
                '\tfind_packages(package_name, pack_list)\n'
                '\tsetup(\n'
                '\t\tname=package_name,\n'
                '\t\tversion=get_app_version(),\n'
                '\t\tkeywords=("pip", package_name, ""),\n'
                '\t\tdescription=package_name,\n'
                '\t\tlong_description="",\n'
                '\t\tlicense="MIT Licence",\n'
                '\t\turl="https://jadehh@live.com",\n'
                '\t\tauthor="jade",\n'
                '\t\tauthor_email="jadehh@live.com",\n'
                '\t\tpackages=pack_list,\n'
                '\t\tpackage_data={"": ["*.so","*.dll"]},\n'
                '\t\tinclude_package_data=True,\n'
                '\t\tplatforms="any",\n'
                '\t\tinstall_requires=[]  # 这个项目需要的第三方库\n'
                '\t\t)').encode("utf-8"))

if __name__ == '__main__':
    import argparse
    import os
    import shutil
    from jade import *
    CreateSavePath("Ouput")
    parser = argparse.ArgumentParser()
    parser.add_argument('--package_name', type=str,
                        default="pyldk")  ## 打包名称
    parser.add_argument('--name', type=str,
                        default="manylinux1_x86_64")  ## 打包名称
    parser.add_argument('--path', type=str,
                        default="pyldk/lib/Linux/x86_64")

    args = parser.parse_args()
    write_setup(args=args)
    os.system("python setup_test.py sdist bdist_wheel")
    shutil.copy("dist/{}-{}-py3-none-any.whl".format(args.package_name, get_app_version()),
                "Ouput/{}-{}-py3-none-{}.whl".format(args.package_name, get_app_version(),args.name))
    shutil.rmtree("dist")
    shutil.rmtree("build")
    shutil.rmtree("{}.egg-info".format(args.package_name))