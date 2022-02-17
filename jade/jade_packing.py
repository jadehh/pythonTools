#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : packing_app.py.py
# @Author   : jade
# @Date     : 2021/11/27 14:25
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from jade import AppRunPath
from jade.jade_tools import CreateSavePath, GetTimeStamp,GetLastDir
from jade.jade_progress_bar import ProgressBar
import os
import shutil
import platform
import subprocess
import uuid

def ui_to_py():
    view_path = "view"
    view_file_list = os.listdir(view_path)
    progressBar = ProgressBar(len(view_file_list))
    for view_name in view_file_list:
        if "ui" in view_name:
            view_file = os.path.join(view_path,view_name)
            os.system("pyuic5 -o {}.py {}".format(view_file.split(".ui")[0],view_file))
        progressBar.update()

def getOperationSystem():
    return platform.system()

def copyPy(args):
    new_src_path = CreateSavePath("new_src")
    if args.is_qt is False:
        src_path_list = ["src"]
        src_import_list = ["src."]
    else:
        src_path_list = ["src","view","view/customView","controller"]
        src_import_list = ["src.","view.","view.customView.","controller."]
    import_list = []
    for src_path in src_path_list:
        for file_name in os.listdir(src_path):
            if "py" in file_name and "init" not in file_name and os.path.isfile(os.path.join(src_path, file_name)):
                if file_name != "samplesMain.py":
                    with open(os.path.join(new_src_path, file_name), "wb") as f1:
                        with open(os.path.join(src_path, file_name), "rb") as f:
                            content_list = str(f.read(), encoding="utf-8").split("\n")
                            for content in content_list:
                                if "import" in content or ("from" in content and "import" in content):
                                    edit = False
                                    for src_import in src_import_list:
                                        if "from" in content:
                                            prefix_list = content.split("from")[1].split("import")[0].split(".")[:-1]
                                            prefix = ""
                                            if len(prefix_list) > 0:
                                                for text in prefix_list:
                                                    prefix =prefix + text + "."
                                            if src_import == prefix.strip():
                                                new_content = content.split(src_import)[0] + \
                                                              content.split(src_import)[1]
                                                f1.write((new_content + '\n').encode("utf-8"))
                                                if new_content not in import_list and "#" not in new_content and \
                                                        new_content[
                                                            0] != " ":
                                                    import_list.append(new_content)
                                                edit = True
                                                break

                                    if edit is False:
                                        f1.write((content + '\n').encode("utf-8"))
                                        if content not in import_list and "#" not in content and content[0] != " ":
                                            import_list.append(content)
                                elif "JadeLog = JadeLogging" in content:
                                    if args.use_jade_log:
                                        update_log = "    JadeLog.INFO('{}-更新时间为:{}',True)\r".format(args.name,GetTimeStamp(),True)
                                        f1.write((content+update_log).encode("utf-8"))
                                    else:
                                        f1.write((content).encode("utf-8"))
                                else:
                                    f1.write((content + "\n").encode("utf-8"))
                else:
                    with open(os.path.join(new_src_path, file_name), "wb") as f1:
                        with open(os.path.join(src_path, file_name), "rb") as f:
                            content_list = str(f.read(), encoding="utf-8").split("\n")
                            for content in content_list:
                                if "import" in content or ("from" in content and "import" in content):
                                    edit = False
                                    for src_import in src_import_list:
                                        if "from" in content:
                                            prefix_list = content.split("from")[1].split("import")[0].split(".")[:-1]
                                            prefix = ""
                                            if len(prefix_list) > 0:
                                                for text in prefix_list:
                                                    prefix =prefix + text + "."
                                            if src_import == prefix.strip():
                                                new_content = content.split(src_import)[0] + \
                                                              content.split(src_import)[1]
                                                f1.write((new_content + '\n').encode("utf-8"))
                                                if new_content not in import_list and "#" not in new_content and \
                                                        new_content[
                                                            0] != " ":
                                                    import_list.append(new_content)
                                                edit = True
                                                break

                                    if edit is False:
                                        f1.write((content + '\n').encode("utf-8"))
                                        if content not in import_list and "#" not in content and content[0] != " ":
                                            import_list.append(content)
                                elif "def main():" in content:
                                    if args.use_jade_log is False:
                                        update_log = "print('#'*20+ '{}-更新时间为:{}' +'#'*20)\r".format(args.name,
                                                                                                     GetTimeStamp())
                                        f1.write((update_log + content).encode("utf-8"))
                                    else:
                                        f1.write(content.encode("utf-8"))
                                else:
                                    f1.write((content + '\n').encode("utf-8"))


    return import_list


def writePy(args):
    import_list = copyPy(args)

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
        try:
            if args.main:
                f.write(args.main.encode("utf-8"))
            else:
                f.write("from samplesMain import main\n"
                "if __name__ == '__main__':\n"
                "    main()\n".encode("utf-8"))
        except:
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
            if type(args.extra_path_list[i]) == tuple:
                bin_path = args.extra_path_list[i][0]
                save_path = args.extra_path_list[i][1]
            else:
                bin_path = args.extra_path_list[i]
                save_path = args.extra_path_list[i]
            if os.path.isdir(bin_path):
                data_list = os.listdir(bin_path)
                for j in range(len(data_list)):
                    file_path = bin_path + "/" + data_list[j]
                    file_path_str = ("'{}'".format(file_path))
                    file_path_list_str = "({},'{}')".format(file_path_str, save_path)
                    if j == len(data_list) - 1 and i == len(args.extra_path_list) - 1:
                        data_str = data_str + file_path_list_str + "]"
                    else:
                        data_str = data_str + file_path_list_str + ","
            else:
                file_path = bin_path
                file_path_str = ("'{}'".format(file_path))
                file_path_list_str = "({},'{}')".format(file_path_str, save_path)
                if  i == len(args.extra_path_list) - 1:
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
                    "          console={},\n"
                    "          icon='{}'\n)\n"
                    "coll = COLLECT(exe2,\n"
                    "          a.binaries,\n"
                    "          a.zipfiles,\n"
                    "          a.datas,\n"
                    "          strip=False,\n"
                    "          upx=True,\n"
                    "          upx_exclude=[],\n"
                    "          name='{}')\n".format(args.app_name, binaries_str, data_str, args.app_name,args.console, icon_path,
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
                    "          console={},\n"
                    "          icon='{}'\n)\n".format(args.app_name, binaries_str, data_str, args.app_name,args.console,icon_path).encode(
                "utf-8"))




def build(args):
    writePy(args)
    if os.path.exists("build"):
        shutil.rmtree("build")
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    os.mkdir("build/")
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

    if len(args.specify_files) > 0:
        progressBar = ProgressBar(len(args.specify_files))
    else:
        progressBar = ProgressBar(len(file_list))

    for file_name in file_list:
        if len(args.specify_files) > 0:
            if file_name in args.specify_files:
                cmd_str = "easycython {}/{}".format( "new_src", file_name)
                result = subprocess.run(cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                progressBar.update()
            else:
                pass
        else:
            cmd_str = "easycython {}/{}".format( "new_src", file_name)
            subprocess.run(cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            progressBar.update()

    build_file_list = os.listdir()
    for build_file in build_file_list:
        if build_file.split(".")[-1] == lib_suffix:

            shutil.copy(build_file,
                        os.path.join(ep_build_path, build_file.split(".")[0] + "." + lib_suffix))
            os.remove(build_file)



    if os.path.exists("src_copy"):
        shutil.rmtree("src_copy")

    if os.path.exists("new_src") is True:
        shutil.rmtree("new_src")

    if os.path.exists("{}.py".format(args.app_name)):
        os.remove("{}.py".format(args.app_name))

    if os.path.exists("{}.spec".format(args.app_name)):
        os.remove("{}.spec".format(args.app_name))
    if args.lib_path:
        if os.path.exists("build"):
            shutil.rmtree("build")


def recursion_dir(file_list,path):
    if os.path.isfile(path):
        file_list.append(path)
    else:
        for file_name in os.listdir(path):
            recursion_dir(file_list,os.path.join(path,file_name))

def get_uuid():
    return "{"+"{"+str(uuid.uuid1())+"}"

def packSetup(args,exec_path):
    file_list = []
    recursion_dir(file_list,exec_path)
    with open("inno_setup.iss",'wb') as f:
        content = "; Script generated by the Inno Setup Script Wizard.\n" \
                  "; SEE THE DOCUMENTATION FOR DETAILS ON CREATING INNO SETUP SCRIPT FILES!\n" \
                  "[Setup]\n" \
                  "; NOTE: The value of AppId uniquely identifies this application.\n" \
                  "; Do not use the same AppId value in installers for other applications.\n" \
                  "; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)\n" \
                  "AppId={}\n" \
                  ";应用名称\n" \
                  "AppName={}\n" \
                  ";应用版本号\n" \
                  "AppVersion={}\n" \
                  ";AppVerName={}\n" \
                  ";应用发布方\n" \
                  "AppPublisher=南京三宝科技有限公司\n" \
                  ";安装目录名称\n" \
                  "DefaultDirName={}\{}\n" \
                  "DefaultGroupName={}\n" \
                  ";安装目录不可选择\n" \
                  "DisableDirPage=auto\n" \
                  ";安装包文件名\n" \
                  "OutputBaseFilename={}\n" \
                  ";压缩包\n" \
                  "Compression=lzma\n" \
                  "SolidCompression=yes\n" \
                  ";安装包图标文件\n" \
                  "SetupIconFile={}\n" \
                  ";许可文件\n" \
                  ";LicenseFile=\n"\
                   "[Files]\n" \
                  ";安装文件\n" \
                .format(get_uuid(),args.name.split("V")[0],args.name.split("V")[-1],args.name,
                                           "{pf64}",args.app_name.split("V")[0],args.app_name.split("V")[0],args.app_name.split("V")[0]+"_setup",
                                           os.path.abspath("icons/app_logo.ico"))

        for file in file_list:
            if len(file.split(exec_path)[-1].split("\\")) > 2:
                path = ""
                for p_file in file.split(exec_path)[-1].split("\\")[:-1]:
                    if p_file:
                        path = path + "\\" + p_file
                cmd_str = 'Source: "{}"; DestDir: "{}\\{}"; Flags: ignoreversion\n'.format(file,'{app}',path)
            else:
                cmd_str = 'Source: "{}"; DestDir: "{}\\{}"; Flags: ignoreversion \n'.format(file,'{app}',"")

            content = content + cmd_str
        content_back = ';[Registry]\n' \
                       ';开机启动\n' \
                       ';Root: HKLM; Subkey: "SOFTWARE\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "MES Monitoring Client"; ValueData: """{}\Client\MES-MonitoringClient.exe"""; Flags: uninsdeletevalue\n' \
                       '[Icons]\n' \
                       'Name: "{}\{}"; Filename: "{}\{}"\n' \
                       'Name: "{}\{}"; Filename: "{}\{}"; Tasks: desktopicon;\n' \
                       '[Tasks]\n' \
                       'Name: "desktopicon"; Description: "{}"; GroupDescription: "{}"; Flags: checkablealone\n' \
                       '[run]\n' \
                       ';两种方法都可以安装服务，上面的可以将服务安装好，但不能直接运行\n' \
                       ';以下的方式可以直接运行，其中有Components:Service;当选中了服务才会安装服务\n' \
                       ';Flags:postinstall点击完成后，才会进行服务的安装，因为在处理RabbitMQ的服务器参数时，不会直接替换参数的\n' \
                       ';安装完成后启动应用\n' \
                       'Filename: "{}\{}"; Description: "{}";Flags:postinstall skipifsilent\n' \
                       '[UninstallRun]\n' \
                       ';卸载时，停止服务并删除服务\n' \
                       ';Filename:{}\sc.exe; Parameters: "stop MESUploadDataService" ; Flags: runhidden; Components:Service\n' \
                       ';Filename: {}\sc.exe; Parameters: "delete MESUploadDataService" ; Flags: runhidden; Components:Service\n' \
                        '[Messages]\n' \
                       ';安装时，windows任务栏提示标题\n' \
                       'SetupAppTitle={}\n' \
                       ';安装时，安装引导标题\n' \
                       'SetupWindowTitle={}\n' \
                       ';在界面左下角加文字\n' \
                       'BeveledLabel=南京三宝科技有限公司\n' \
                       ';卸载对话框说明\n' \
                       'ConfirmUninstall=您真的想要从电脑中卸载 %1 吗?%n%n按 [是] 则完全删除 %1 以及它的所有组件;%n按 [否]则让软件继续留在您的电脑上.\n' \
                       ';[Types]\n' \
                       ';Name: "normaltype"; Description: "Normal Setup"\n' \
                       ';Name: "custom";     Description: "Custom Installation"; Flags: iscustom\n' \
                       ';[Components]\n' \
                       ';Name: "Client";     Description: "应用界面";  Types: normaltype custom\n' \
                       '[Code]\n' \
                       '//设置界面文字颜色\n' \
                       'procedure InitializeWizard();\n' \
                       'begin\n' \
                       '//WizardForm.WELCOMELABEL1.Font.Color:= clGreen;//设置开始安装页面第一段文字的颜色为绿色\n' \
                       '//WizardForm.WELCOMELABEL2.Font.Color:= clOlive;//设置开始安装页面第二段文字的颜色为橄榄绿\n' \
                       '//WizardForm.PAGENAMELABEL.Font.Color:= clRed;//设置许可协议页面第一段文字的颜色为红色\n' \
                       '//WizardForm.PAGEDESCRIPTIONLABEL.Font.Color:= clBlue; //设置许可协议页面第二段文字的颜色为蓝色\n' \
                       'WizardForm.MainPanel.Color:= clWhite;//设置窗格的颜色为白色\n' \
                       'end;\n' \
                       '//卸载后打开网址\n' \
                       '//procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);\n' \
                       '//var\n' \
                       '//  ErrorCode: Integer;\n' \
                       '//begin\n' \
                       '//  case CurUninstallStep of\n' \
                       '//   usUninstall:\n' \
                       '//     begin\n' \
                       '// 正在卸载\n' \
                       '//    end;\n' \
                       '//  usPostUninstall:\n' \
                       '//    begin\n' \
               .format("{app}",
                       "{group}",args.name,"{app}",args.app_name+".exe",
                                       "{commondesktop}",args.name,"{app}",args.app_name+".exe",
                                       "{cm:CreateDesktopIcon}","{cm:AdditionalIcons}",
                                       "{app}",args.app_name+".exe","{cm:LaunchProgram,ContainerOCR}",
                                       "{sys}","{sys}",
                       args.app_name.split("V")[0]+"_setup",args.app_name.split("V")[0]+"_setup")
        content = content+content_back
        f.write(content.encode("gbk"))

def packAppImage(args):
    save_path = CreateSavePath(os.path.join("tmp"))
    save_bin_path = CreateSavePath(os.path.join(save_path, "usr/bin/"))
    if args.full is False:
        os.system("cp -r dist/{}/* {}".format(args.app_name, save_bin_path))
        ## 需要在AppRun中添加环境变量
    else:
        # 打包成一个包环境变量就没了
        save_lib_path = CreateSavePath(os.path.join(save_path, "usr/lib/"))
        if args.extra_path_list:
            for i in range(len(args.extra_path_list)):
                lib_path = args.extra_path_list[i]
                if type(lib_path) == tuple:
                    lib_path = lib_path[0]
                for lib_name in os.listdir(lib_path):
                    if "lib" in lib_name:
                        shutil.copy(os.path.join(lib_path, lib_name), os.path.join(save_lib_path, lib_name))
        os.system("cp -r dist/{} {}".format(args.app_name, save_bin_path))

    with open(AppRunPath, "r") as f:
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
    cmd_str = "pyinstaller  {}.spec  --additional-hooks-dir hooks".format(args.app_name)
    os.system(cmd_str)
    save_path = CreateSavePath(os.path.join("releases",args.name))
    if os.path.exists("{}/{}".format(getOperationSystem(), save_path)) is True:
        shutil.rmtree("{}/{}".format(getOperationSystem(), save_path))
    save_bin_path = CreateSavePath(os.path.join(save_path, getOperationSystem()))
    # copy_dir("config", save_bin_path)
    if args.lib_path:
        copy_dir(args.lib_path, save_bin_path)
    if "Windows" == getOperationSystem():
        if args.full is False:
            os.system("xcopy dist\\{} {} /s/y".format(args.app_name,save_bin_path))
        else:
            shutil.copy("dist\\{}.exe".format(args.app_name), "{}/".format(save_bin_path))
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
        parser.add_argument('--extra_sys_list', type=list,
                            default=[])  ## 需要额外打包的路径
        parser.add_argument('--extra_path_list', type=list,
                            default=["qss",("data","xpinyin"),"resources"])
    else:
        parser.add_argument('--extra_sys_list', type=list,
                            default=['sys.path.append("/usr/local/conta_detect-1.0/python_lib")'])  ## sys.path.append需要额外打包的路径

        parser.add_argument('--extra_path_list', type=list,
                            default=[])  ## 需要额外打包的路径
    parser.add_argument('--use_jade_log', type=bool,
                        default=False) ##是否使用JadeLog
    parser.add_argument('--full', type=bool,
                        default=False)  ## 打包成一个完成的包
    parser.add_argument('--console', type=str,
                        default="False")  ## 是否显示命令行窗口,只针对与Windows有效
    parser.add_argument('--app_name', type=str,
                        default="container_ocr_programV2.4.3")  ##需要打包的文件名称
    parser.add_argument('--name', type=str,
                        default="箱号识别服务前端配置界面V2.4.3")  ##需要打包的文件名称
    parser.add_argument('--appimage', type=bool,
                        default=False)  ## 是否打包成AppImage
    parser.add_argument('--lib_path', type=str, default="container_ocr_program_lib64")  ## 是否lib包分开打包
    parser.add_argument('--is_qt', type=bool, default=True)  ## qt 会将controller view src 都进行编译
    parser.add_argument('--specify_files',type=list,default=["customTextBrowser.py"]) ## 指定编译的文件

    args = parser.parse_args()
    # ui_to_py()
    build(args)
    packAPP(args)

