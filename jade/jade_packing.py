#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : jade_packing_windows_app.py.py
# @Author   : jade
# @Date     : 2021/9/22 11:54
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from jade import CreateSavePath, GetTimeStamp
import os
import shutil
import platform

def getOperationSystem():
    return platform.system()

def copyPy():
    new_src_path = CreateSavePath("new_src")
    src_path = "src"
    import_list = []
    for file_name in os.listdir(src_path):
        if "py" in file_name and "init" not in file_name and os.path.isfile(os.path.join(src_path, file_name)):
            if file_name != "samplesMain.py":
                with open(os.path.join(new_src_path, file_name), "wb") as f1:
                    with open(os.path.join(src_path, file_name), "rb") as f:
                        content_list = str(f.read(), encoding="utf-8").split("\n")
                        for content in content_list:
                            if "import" in content or ("from" in content and "import" in content):

                                if "src." in content:
                                    new_content = content.split("src.")[0] + content.split("src.")[1]
                                    f1.write((new_content + '\n').encode("utf-8"))
                                    if new_content not in import_list and "#" not in new_content and new_content[
                                        0] != " ":
                                        import_list.append(new_content)
                                else:
                                    f1.write((content + '\n').encode("utf-8"))
                                    if content not in import_list and "#" not in content and content[0] != " ":
                                        import_list.append(content)
                            else:
                                f1.write((content + "\n").encode("utf-8"))
            else:
                with open(os.path.join(new_src_path, file_name), "wb") as f1:
                    with open(os.path.join(src_path, file_name), "rb") as f:
                        content_list = str(f.read(), encoding="utf-8").split("\n")
                        for content in content_list:
                            if "import" in content or ("from" in content and "import" in content):

                                if "src." in content:
                                    new_content = content.split("src.")[0] + content.split("src.")[1]
                                    f1.write((new_content + '\n').encode("utf-8"))
                                    if new_content not in import_list and "#" not in new_content and new_content[
                                        0] != " ":
                                        import_list.append(new_content)
                                else:
                                    f1.write((content + '\n').encode("utf-8"))
                                    if content not in import_list and "#" not in content and content[0] != " ":
                                        import_list.append(content)
                            elif "main" in content:
                                f1.write((
                                                     content + '\n    JadeLog.INFO("#####################版本更新时间为:{}#####################")\r'.format(
                                                 GetTimeStamp())).encode("utf-8"))
                            else:
                                f1.write((content + '\n').encode("utf-8"))

    return import_list


def writePy(args):
    import_list = copyPy()

    with open("{}.py".format(args.app_name), "wb") as f:
        f.write("import sys\n"
                "import os\n"
                "if getattr(sys, 'frozen', False): #是否Bundle Resource\n"
                "    base_path = sys._MEIPASS\n"
                "else:\n"
                "    base_path = os.path.abspath('.')\n"
                "sys.path.append('{}')\n"
                "sys.path.append(os.path.join(base_path,'build/encryption'))\n".format(args.lib_path).encode("utf-8"))

        for extra_sys_path in args.extra_sys_list:
            f.write(extra_sys_path.encode("utf-8") + "\n".encode("utf-8"))
        for import_src in import_list:
            f.write(import_src.encode("utf-8") + "\n".encode("utf-8"))

        f.write("from samplesMain import main\n"
                "if __name__ == '__main__':\n"
                "    main()\n".encode("utf-8"))


def writeSpec(args):
    data_str = "datas=["
    if args.lib_path:
        pass
    else:
        file_list = os.listdir("build/encryption")
        for i in range(len(file_list)):
            file_path = os.path.join("build/encryption", file_list[i])
            file_path_str = ("'{}'".format(file_path))
            file_path_list_str = "({},'.')".format(file_path_str)
            data_str = data_str + file_path_list_str + ","

    if len(args.extra_path_list) == 0:
        data_str = data_str + "]"
    else:
        for i in range(len(args.extra_path_list)):
            bin_path = args.extra_path_list[i]
            data_list = os.listdir(bin_path)
            for j in range(len(data_list)):
                file_path = bin_path + "/" + data_list[j]
                file_path_str = ("'{}'".format(file_path))
                file_path_list_str = "({},'{}')".format(file_path_str, bin_path)
                if j == len(data_list) - 1 and i == len(args.extra_path_list) - 1:
                    data_str = data_str + file_path_list_str + "]"
                else:
                    data_str = data_str + file_path_list_str + ","

    binaries_str = "binaries=["
    icon_list = os.listdir("icons")
    for i in range(len(icon_list)):
        if i == len(icon_list) - 1:
            binaries_str = binaries_str + "('icons/{}','{}')]".format(icon_list[i], "icons")
        else:
            binaries_str = binaries_str + "('icons/{}','{}'),".format(icon_list[i], "icons")
    icon_path = "icons/app_logo.ico"
    if args.full is False:
        with open("{}.spec".format(args.app_name), "wb") as f:
            f.write("block_cipher = None\n"
                    "a = Analysis(['{}.py'],\n"
                    "             pathex=[''],\n"
                    "             {},\n"
                    "             {},\n"
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
                    "exe2 = EXE(pyz,\n"
                    "          a.scripts,\n"
                    "          [],\n"
                    "          exclude_binaries=True,\n"
                    "          name='{}',\n"
                    "          debug=False,\n"
                    "          bootloader_ignore_signals=False,\n"
                    "          strip=False,\n"
                    "          upx=True,\n"
                    "          console=False,\n"
                    "          icon='{}'\n)\n"
                    "coll = COLLECT(exe2,\n"
                    "          a.binaries,\n"
                    "          a.zipfiles,\n"
                    "          a.datas,\n"
                    "          strip=False,\n"
                    "          upx=True,\n"
                    "          upx_exclude=[],\n"
                    "          name='{}')\n".format(args.app_name, binaries_str, data_str, args.app_name, icon_path,
                                                    args.app_name).encode("utf-8"))
    else:
        with open("{}.spec".format(args.app_name), "wb") as f:
            f.write("block_cipher = None\n"
                    "a = Analysis(['{}.py'],\n"
                    "             pathex=[''],\n"
                    "             {},\n"
                    "             {},\n"
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
                    "exe1 = EXE(pyz,\n"
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
                    "          console=True,\n"
                    "          icon='{}'\n)\n".format(args.app_name, binaries_str, data_str, args.app_name,
                                                      icon_path).encode(
                "utf-8"))


def build(args):
    writePy(args)
    ID = int(args.ID)
    if os.path.exists("build"):
        shutil.rmtree("build")
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    os.mkdir("build/")
    print("ID = {}".format(ID))

    if args.lib_path:
        ep_build_path = args.lib_path
    else:
        ep_build_path = "build/encryption/"

    if os.path.exists(ep_build_path):
        shutil.rmtree(ep_build_path)
    os.mkdir(ep_build_path)
    file_list = os.listdir("new_src")
    bin_suffix = ""

    if getOperationSystem() == "Windows":
        bin_suffix = ".exe"
    if getOperationSystem() == "Windows":
        lib_suffix = "pyd"
    else:
        lib_suffix = "so"
    for file_name in file_list:
        os.system("{}/easycython{} {}/{}".format(args.python_path, bin_suffix, "new_src", file_name))

    build_file_list = os.listdir()
    for build_file in build_file_list:
        if build_file.split(".")[-1] == lib_suffix:
            if ID == 0:
                shutil.copy(build_file,
                            os.path.join(ep_build_path, build_file.split(".")[0] + "." + lib_suffix))
                os.remove(build_file)
            else:
                os.system(
                    "/home/jade/SoftWare/加密狗软件/Linux/VendorTools/Envelope/linuxenv -v:/home/jade/RGMGT.hvc -f:{} {} {}".format(
                        ID, build_file,
                        os.path.join(ep_build_path, build_file.split(".")[0] + ".pyd")))

    if os.path.exists("src_copy"):
        shutil.rmtree("src_copy")

    if os.path.exists("new_src") is True:
        shutil.rmtree("new_src")

    if os.path.exists("{}.py".format(args.app_name)):
        os.remove("{}.py".format(args.app_name))

    if os.path.exists("{}.spec".format(args.app_name)):
        os.remove("{}.spec".format(args.app_name))


def packAppImage(args):
    save_path = CreateSavePath(os.path.join("tmp"))
    save_bin_path = CreateSavePath(os.path.join(save_path, "usr/bin/"))
    if args.full is False:
        os.system("cp -r dist/{}/* {}".format(args.app_name, save_bin_path))
        ## 需要在AppRun中添加环境变量
    else:
        # 打包成一个包环境变量就没了
        save_lib_path = CreateSavePath(os.path.join(save_path, "usr/lib/"))
        for lib_path in args.extra_path_list:
            for lib_name in os.listdir(lib_path):
                if "lib" in lib_name:
                    shutil.copy(os.path.join(lib_path, lib_name), os.path.join(save_lib_path, lib_name))
        os.system("cp -r dist/{} {}".format(args.app_name, save_bin_path))

    with open("AppRun", "r") as f:
        conetent_list = f.read().split("\n")
        for content in conetent_list:
            with open(os.path.join(save_path, "AppRun"), "a", encoding="utf-8") as f:
                f.write(content + "\n")
    shutil.copy("icons/app_logo.png", save_path)
    with open(os.path.join(save_path, args.app_name + ".desktop"), "w", encoding="utf-8") as f:
        f.write("[Desktop Entry]\n"
                "Version=1.0\n"
                "Name={}\n"
                "Type=Application\n"
                "Categories=Qt;\n"
                "Terminal=false\n"
                "Icon=app_logo\n"
                "Exec={} %u\n"
                "MimeType=x-scheme-handler/qv2ray;\n"
                "X-AppImage-Version=912fe1b\n\n\n"
                "Name[zh_CN]={}".format(args.app_name, args.app_name, args.app_name))
    print("{}/appimagetool-x86_64.AppImage {} {}.AppImage".format(os.path.expanduser("~"), "tmp", args.app_name))
    os.system("{}/appimagetool-x86_64.AppImage {} {}.AppImage".format(os.path.expanduser("~"), "tmp", args.app_name))
    os.system("chmod +x  {}.AppImage".format(args.app_name))
    return "{}.AppImage".format(args.app_name)


def copy_dir(source_dir, save_path):
    try:
        shutil.rmtree("{}/{}".format(save_path,source_dir))
    except:
        pass
    try:
        shutil.copytree(source_dir, "{}/{}".format(save_path,source_dir))
    except :
        pass



def packAPP(args):
    writePy(args)
    writeSpec(args)
    cmd_str = "{}/pyinstaller  {}.spec ".format(args.python_path, args.app_name)
    os.system(cmd_str)
    save_path = CreateSavePath(os.path.join("releases",args.name))
    if os.path.exists("{}/{}".format(getOperationSystem(), save_path)) is True:
        shutil.rmtree("{}/{}".format(getOperationSystem(), save_path))
    save_bin_path = CreateSavePath("{}/{}".format(save_path, getOperationSystem()))
    copy_dir("config", save_bin_path)
    if args.lib_path:
        copy_dir(args.lib_path, save_bin_path)
    if "Windows" == getOperationSystem():
        shutil.copy("dist/{}.exe".format(args.app_name), "{}/".format(save_bin_path))
    else:
        if args.appimage:
            app_name = packAppImage(args)
            shutil.copy(app_name, "{}/".format(save_bin_path))
        else:
            shutil.copy("dist/{}".format(args.app_name), "{}/".format(save_bin_path))
    if os.path.exists("{}.py".format(args.app_name)):
        os.remove("{}.py".format(args.app_name))
    if os.path.exists("{}.spec".format(args.app_name)):
        os.remove("{}.spec".format(args.app_name))

    if os.path.exists("new_src") is True:
        shutil.rmtree("new_src")

    if os.path.exists("build"):
        shutil.rmtree("build")

    if os.path.exists(args.lib_path):
        shutil.rmtree(args.lib_path)

    if os.path.exists("dist"):
        shutil.rmtree("dist")

    if os.path.exists("{}.AppImage".format(args.app_name)):
        os.remove("{}.AppImage".format(args.app_name))

    if os.path.exists("tmp"):
        shutil.rmtree("tmp")



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    if getOperationSystem() == "Windows":
        parser.add_argument("--python_path", type=str,
                            default=r"C:\Users\Administrator\.virtualenvs\SuzhouPark5GAI-ioO_3PBQ\Scripts/")
    else:
        parser.add_argument("--python_path", type=str,
                            default="/home/jade/.local/share/virtualenvs/SuzhouPark5GAI-oaurvAjI//bin/")
    parser.add_argument('--extra_path_list', type=list,
                        default=["bin/{}/".format(getOperationSystem())])  ## 需要额外打包的路径
    parser.add_argument('--ID', type=str,
                        default="0")
    parser.add_argument('--full', type=bool,
                        default=True)  ## 打包成一个完成的包
    parser.add_argument('--app_name', type=str,
                        default="SuzhouDCDetServiceV1.0")  ##需要打包的文件名称
    parser.add_argument('--name', type=str,
                        default="苏州电子围网检测服务V1.0")  ##需要打包的文件名称
    parser.add_argument('--appimage', type=bool,
                        default=False)  ## 是否打包成AppImage
    parser.add_argument('--lib_path', type=str, default="dc_detect_service_lib64")  ## 是否lib包分开打包
    parser.add_argument('--extra_sys_list', type=list,
                        default=['sys.path.append("/usr/local/suzhou_park-1.0/python_lib/")', ])  ## 需要额外打包的路径

    args = parser.parse_args()

    build(args)
    # packAPP(args)
