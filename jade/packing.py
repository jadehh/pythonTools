#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : packing.py
# @Author   : jade
# @Date     : 2021/5/15 16:52
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from jade import CreateSavePath,GetTimeStamp
import os
import shutil


def copyPy():
    new_src_path = CreateSavePath("new_src")
    src_path = "src"
    import_list = []
    for file_name in os.listdir(src_path):
        if "py" in file_name and "init" not in file_name and os.path.isfile(os.path.join(src_path, file_name)):
            if "Main" not in file_name:
                with open(os.path.join(new_src_path, file_name), "w") as f1:
                    with open(os.path.join(src_path, file_name), "rb") as f:
                        content_list = str(f.read(), encoding="utf-8").split("\n")
                        for content in content_list:
                            if "import" in content or ("from" in content and "import" in content):

                                if "src." in content:
                                    new_content = content.split("src.")[0] + content.split("src.")[1]
                                    f1.write(new_content + '\n')
                                    if new_content not in import_list and "#" not in new_content and new_content[
                                        0] != " ":
                                        import_list.append(new_content)
                                else:
                                    f1.write(content + '\n')
                                    if content not in import_list and "#" not in content and content[0] != " ":
                                        import_list.append(content)
                            else:
                                f1.write(content + "\n")
            else:
                with open(os.path.join(new_src_path, file_name), "w") as f1:
                    with open(os.path.join(src_path, file_name), "rb") as f:
                        content_list = str(f.read(), encoding="utf-8").split("\n")
                        for content in content_list:
                            if "import" in content or ("from" in content and "import" in content):

                                if "src." in content:
                                    new_content = content.split("src.")[0] + content.split("src.")[1]
                                    f1.write(new_content + '\n')
                                    if new_content not in import_list and "#" not in new_content and new_content[
                                        0] != " ":
                                        import_list.append(new_content)
                                else:
                                    f1.write(content + '\n')
                                    if content not in import_list and "#" not in content and content[0] != " ":
                                        import_list.append(content)
                            elif "main" in content:
                                f1.write(
                                    content + '    JadeLog.INFO("#####################版本更新时间为:{}#####################")\r'.format(
                                        GetTimeStamp()))
                            else:
                                f1.write(content + "\n")

    return import_list


def writePy(app_name):
    import_list = copyPy()
    with open("{}.py".format(app_name), "w") as f:
        f.write("import sys\n"
                "sys.path.append('lib/')\n"
                "sys.path.append('/usr/local/lib/')\n"
                "from samplesMain import main\n")
        for import_src in import_list:
            f.write(import_src)

        f.write("if __name__ == '__main__':\n"
                "    main()\n")



def writeSpec(app_name):
    with open("{}.spec".format(app_name), "w") as f:
        f.write("# -*- mode: python ; coding: utf-8 -*-\n"
                "block_cipher = None\n"
                "a = Analysis(['{}.py'],\n"
                "pathex=[],\n"
                "binaries=[],\n"
                "datas=[],\n"
                " hiddenimports=[],\n"
                "hookspath=[],\n"
                "runtime_hooks=[],\n"
                "excludes=[],\n"
                "win_no_prefer_redirects=False,\n"
                " win_private_assemblies=False,\n"
                "cipher=block_cipher,\n"
                "noarchive=False)\n"
                "pyz = PYZ(a.pure, a.zipped_data,\n"
                "cipher=block_cipher)\n"
                "exe = EXE(pyz,\n"
                " a.scripts,\n"
                "a.binaries,\n"
                "a.zipfiles,\n"
                " a.datas,\n"
                "[],\n"
                "name='{}',\n"
                "debug=False,\n"
                "bootloader_ignore_signals=False,\n"
                "strip=False,\n"
                " upx=True,\n"
                "upx_exclude=[],\n"
                "runtime_tmpdir=None,\n"
                "console=True )\n".format(app_name, app_name))


def build(args):
    writePy(args.app_name)
    ID = int(args.ID)
    if os.path.exists("build"):
        shutil.rmtree("build")
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    os.mkdir("build/")
    os.mkdir("build/{}_加密狗ID={}/".format(args.name, args.ID))
    print("ID = {}".format(ID))
    build_path = "build/lib.linux-x86_64-3.6"
    tmp_path = "build/temp.linux-x86_64-3.6"
    ep_build_path = "build/{}_加密狗ID={}/lib".format(args.name, args.ID)

    if os.path.exists(build_path):
        shutil.rmtree(build_path)

    if os.path.exists(ep_build_path):
        shutil.rmtree(ep_build_path)
    os.mkdir(ep_build_path)
    os.system("{}/python setup.py build_ext".format(args.python_path))
    build_file_list = os.listdir(os.path.join(build_path))
    for build_file in build_file_list:
        if build_file[-3:] == ".so":
            if ID == 0:
                shutil.copy(os.path.join(build_path, build_file),
                            os.path.join(ep_build_path, build_file.split(".")[0] + ".so"))
            else:
                os.system(
                    "/home/jade/SoftWare/加密狗软件/Linux/VendorTools/Envelope/linuxenv -v:/home/jade/RGMGT.hvc -f:{} {} {}".format(
                        ID, os.path.join(build_path, build_file),
                        os.path.join(ep_build_path, build_file.split(".")[0] + ".so")))

    shutil.rmtree("src_copy")
    shutil.rmtree(build_path)
    shutil.rmtree(tmp_path)
    if os.path.exists("new_src") is True:
        shutil.rmtree("new_src")


def packing(args):
    writeSpec(args.app_name)
    cmd_str = "{}/pyinstaller -F {}.spec".format(args.python_path, args.app_name)
    os.system(cmd_str)
    if os.path.exists(args.name) is True:
        shutil.rmtree(args.name)
    os.mkdir(args.name)
    os.mkdir(os.path.join(args.name, "lib"))
    file_list = os.listdir("dist")
    ep_build_path = "build/{}_加密狗ID={}/lib".format(args.name, args.ID)
    for build_file in os.listdir(ep_build_path):
        shutil.copy(os.path.join(ep_build_path, build_file), os.path.join(args.name, "lib/{}").format(build_file))
    for file_name in file_list:
        if os.path.isdir(os.path.join("dist", file_name)):
            shutil.copy(os.path.join(os.path.join("dist", file_name), file_name),
                        os.path.join(args.name, "{}".format(file_name)))
        else:
            shutil.copy(os.path.join("dist", file_name), os.path.join(args.name, "{}".format(file_name)))
    if os.path.exists("build") is True:
        shutil.rmtree("build")
    if os.path.exists("dist") is True:
        shutil.rmtree("dist")
    if os.path.exists("{}.spec".format(args.app_name)) is True:
        os.remove("{}.spec".format(args.app_name))
    if os.path.exists("{}.py".format(args.app_name)) is True:
        os.remove("{}.py".format(args.app_name))
    if os.path.exists("new_src") is True:
        shutil.rmtree("new_src")