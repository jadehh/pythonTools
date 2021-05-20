#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : jade_packing.py
# @Author   : jade
# @Date     : 2021/5/15 16:52
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from jade import CreateSavePath, GetTimeStamp
import os
import shutil


def copyPy():
    new_src_path = CreateSavePath("new_src")
    src_path = "src"
    import_list = []
    for file_name in os.listdir(src_path):
        if "py" in file_name and "init" not in file_name and os.path.isfile(os.path.join(src_path, file_name)):
            if file_name != "samplesMain.py":
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
                                    content)
                            else:
                                f1.write(content + "\n")

    return import_list


def writePy(app_name):
    import_list = copyPy()
    with open("{}.py".format(app_name), "w") as f:
        f.write("import sys\n"
                "sys.path.append('build/encryption')\n"
                "from samplesMain import main\n")
        for import_src in import_list:
            f.write(import_src)

        f.write("if __name__ == '__main__':\n"
                "    main()\n")


def writeSpec(app_name):
    build_path = "build/encryption/"
    data_str = "datas=["
    file_list = os.listdir(build_path)
    for i in range(len(file_list)):
        file_path = os.path.join(build_path,file_list[i])
        file_path_str = ("'{}'".format(file_path))
        file_path_list_str = "({},'.')".format(file_path_str)
        if i == len(file_list) -1 :
            data_str = data_str + file_path_list_str
        else:
            data_str = data_str + file_path_list_str + ","

    with open("{}.spec".format(app_name), "w") as f:
        f.write("block_cipher = None\n"
                "a = Analysis(['{}.py'],\n"
                "             pathex=[''],\n"
                "             binaries=[],\n"
                "             {}],\n"
                "             hiddenimports=[],\n"
                "             hookspath=[],\n"
                "             runtime_hooks=[],\n"
                "             excludes=[],\n"
                "             win_no_prefer_redirects=False,\n"
                "             win_private_assemblies=False,\n"
                "             cipher=block_cipher,\n"
                "             noarchive=False)\n"
                "pyz = PYZ(a.pure, a.zipped_data,\n"
                "             cipher=block_cipher)\n"
                "exe = EXE(pyz,\n"
                "          a.scripts,\n"
                "          a.binaries,\n"
                "          a.zipfiles,\n"
                "          a.datas,\n"
                "          [],\n"
                "          name='{}',\n"
                "          debug=False,\n"
                "          bootloader_ignore_signals=False,\n"
                "          strip=False,\n"
                "          upx=True,\n"
                "          upx_exclude=[],\n"
                "          runtime_tmpdir=None,\n"
                "          console=True )\n".format(app_name,data_str,app_name))


def build(args):
    writePy(args.app_name)
    ID = int(args.ID)
    if os.path.exists("build"):
        shutil.rmtree("build")
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    os.mkdir("build/")
    print("ID = {}".format(ID))
    build_path = "build/lib.linux-x86_64-3.6"
    tmp_path = "build/temp.linux-x86_64-3.6"
    ep_build_path = "build/encryption/"

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


def packAPP(args):
    writeSpec(args.app_name)
    cmd_str = "{}/pyinstaller -F {}.spec".format(args.python_path, args.app_name)
    os.system(cmd_str)
    if os.path.exists(args.name):
        shutil.rmtree(args.name)
    save_path = CreateSavePath(os.path.join("tmp"))
    save_bin_path = CreateSavePath(os.path.join(save_path, "usr/bin/"))
    shutil.copy("dist/{}".format(args.app_name), save_bin_path)
    with open("AppRun", "r") as f:
        conetent_list = f.read().split("\n")
        for content in conetent_list:
            with open(os.path.join(save_path, "AppRun"), "a", encoding="utf-8") as f:
                f.write(content + "\n")
    shutil.copy("icons/samples.png", save_path)
    with open(os.path.join(save_path, args.app_name + ".desktop"), "w", encoding="utf-8") as f:
        f.write("[Desktop Entry]\n"
                "Version=1.0\n"
                "Name={}\n"
                "Type=Application\n"
                "Categories=Qt;\n"
                "Terminal=false\n"
                "Icon=samples\n"
                "Exec={} %u\n"
                "MimeType=x-scheme-handler/qv2ray;\n"
                "X-AppImage-Version=912fe1b\n\n\n"
                "Name[zh_CN]={}".format(args.app_name, args.app_name, args.app_name))
    os.system("{}/appimagetool-x86_64.AppImage {} {}.AppImage".format(os.path.expanduser("~"), "tmp", args.app_name))
    os.system("chmod +x  {}.AppImage".format(args.app_name))
    App_dir = CreateSavePath(args.name)
    shutil.copy("{}.AppImage".format(args.app_name),App_dir)
    if os.path.exists("tmp"):
        shutil.rmtree("tmp")
    if os.path.exists("build"):
        shutil.rmtree("build")
    if os.path.exists("dist"):
        shutil.rmtree("dist")

    if os.path.exists("{}.AppImage".format(args.app_name)):
        os.remove("{}.AppImage".format(args.app_name))

    if os.path.exists("{}.py".format(args.app_name)):
        os.remove("{}.py".format(args.app_name))

    if os.path.exists("{}.spec".format(args.app_name)):
        os.remove("{}.spec".format(args.app_name))









if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_path", type=str,
                        default="/home/jade/.local/share/virtualenvs/container_ocr-OdTDZGFh/bin//")
    parser.add_argument('--ID', type=str,
                        default="0")
    parser.add_argument('--app_name', type=str,
                        default="ContainerOCR")  ##需要打包的文件名称
    parser.add_argument('--name', type=str,
                        default="箱号识别服务前端配置程序")  ##需要打包的文件名称

    args = parser.parse_args()
    build(args)
    packAPP(args)