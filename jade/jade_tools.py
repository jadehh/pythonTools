#coding=utf-8
import datetime
import sys
import time
import cv2
import os
#import xlrd
import os.path as ops
from jade import *
class ProcessBar():
    def __init__(self):
        self.count = 0
        self.index = 0
        self.number = 30
        self.start_time = 0




class ProTxt():
    def __init__(self):
        self.name = None
        self.label = None
        self.displayname = None

#读取prototxt文件
def ReadProTxt(file_path,id=True):
    with open(file_path,'r') as f:
        result = f.read().split("\n")
    protxt = []
    dics = {}
    class_names = []
    for i in range(len(result)):
        if i % 5 == 1:
            name = result[i].split('name: ')[1].split('"')[1]
            class_names.append(name)
        elif i % 5 == 2:
            label = int(result[i].split("label: ")[1])
        elif i % 5 == 3:
            display_name = result[i].split("display_name: ")[1].split('"')[1]
        elif i % 5 == 4 and i > 5:
            if id:
                dic = {"name": name, "display_name": display_name}
                # print(dic)
                dics.update({label: dic})
            else:
                dic = {"id": label, "display_name": display_name}
                dics.update({name: dic})

    return dics,class_names

def GetSysPath():
    syspath = ""
    for sys_path in sys.path:
        if "site-packages" in sys_path in sys_path and os.path.isdir(sys_path):
            syspath = sys_path.split("site-packages")[0] + "site-packages/"
            break
        if  'dist-packages' in sys_path in sys_path and os.path.isdir(sys_path):
            syspath = sys_path.split("dist-packages")[0] + "dist-packages/"
    return syspath


#合并文件路径
def OpsJoin(path1,path2):
    return ops.join(path1,path2)

#返回上一层目录
def GetPreviousDir(savepath):

    return os.path.dirname(savepath)
#返回最后一层的目录
def GetLastDir(savepath):
    return os.path.basename(savepath)


#获取文件夹下，后缀为.的文件
def GetFilesWithLastNamePath(dir,lastname):
    imagename_list = os.listdir(dir)
    image_list = []
    for image_name in imagename_list:
        last = "."+image_name.split(".")[-1]
        if last == lastname:
            image_list.append(os.path.join(dir,image_name))
    return (image_list)

#获取一个文件夹下所有的图片列表
def GetAllImagesNames(dir):
    imagename_list = os.listdir(dir)
    image_list = []
    for image_name in imagename_list:
        if image_name[-4:].lower == ".jpg" or image_name[-4:].lower() == ".png":
            image_list.append(image_name)
    return (image_list)

#获取一个文件夹下所有的图片路径
def GetAllImagesPath(dir):
    imagename_list = os.listdir(dir)
    image_list = []
    for image_name in imagename_list:
        if image_name[-4:].lower() == ".jpg" or image_name[-4:].lower() == ".png":
            image_list.append(OpsJoin(dir,image_name))
    return (image_list)
#获取今天的日期
def GetToday():
    now = datetime.datetime.now()
    otherStyleTime = now.strftime("%Y-%m-%d %H:%M:%S")
    pathname = otherStyleTime.split(" ")[0]
    return pathname
#获取当前的时间
def GetHourTime():
    now = datetime.datetime.now()
    otherStyleTime = now.strftime("%Y-%m-%d %H-%M-%S")
    pathname = otherStyleTime.split(" ")[1]
    return pathname


def GetTime():
    now = datetime.datetime.now()
    otherStyleTime = now.strftime("%Y-%m-%d-%H-%M-%S")
    pathname = otherStyleTime
    data_ms = datetime.datetime.now().microsecond / 1000
    time_stamp = "%s-%03d" % (pathname, data_ms)
    return time_stamp


def GetTimeStamp():
    now = datetime.datetime.now()
    otherStyleTime = now.strftime("%Y-%m-%d %H:%M:%S")
    pathname = otherStyleTime
    return pathname











#欧氏距离计算
def Eucliden_Distance(feature1,feature2):
    dist = np.sqrt(np.sum(np.square(np.subtract(feature1, feature2))))
    return dist


#不换行输出并且打印输出的东西
def NoLinePrint(output,processbar=None):
    if processbar is None:
        sys.stdout.write('\r>> ' + time.strftime("%Y-%m-%d %H:%M", time.localtime()) + ": " + str(output))
        sys.stdout.flush()
    else:
        processbar.index = processbar.index + 1
        num = (int(processbar.index*processbar.number/processbar.count))
        complete = "|"+'█'*num + " " *(processbar.number-num)+"|"
        if processbar.start_time == 0:
            use_time = "[please correct start_time]"
        else:
            use_time = " [use time : %f]" %(time.time()-processbar.start_time)
        pec = float((processbar.index*100)/processbar.count)
        sys.stdout.write('\r>> ' + time.strftime("%Y-%m-%d %H:%M", time.localtime()) + ": " + str(output)+ complete+" %d / %d"%(processbar.index,processbar.count)+use_time)
        sys.stdout.flush()

def Line_Print(output):
    print(time.strftime(">> "+"%Y-%m-%d %H:%M", time.localtime())+":"+" "+output)

#秒格式化输出
def Seconds_To_Hours(seconds):
    seconds = int(seconds)
    m,s = divmod(seconds,60)
    h,m = divmod(m,60)
    return ("%02d:%02d:%02d"%(h,m,s))




#改变路径/home/jade/
def Change_Dir(dir):
    change_cut = (os.getcwd()).split("/")[1:3]
    changedir = ""
    for path in change_cut:
        changedir = changedir+"/" + path
    dic_cut = dir.split("/")[3:]
    for dic_cange in dic_cut:
        changedir = changedir + "/" + dic_cange
    print(changedir)
    return changedir



def GetRootPath():
    return os.path.expanduser("~") + "/"


#xml准换成prototxt
def XMLTOPROTXT(xlsx_path,protxt_name="ThirtyTypes"):
    data = xlrd.open_workbook(xlsx_path)
    table_count = len(data.sheets())
    index = 0
    for i in range(table_count):
        table = data.sheets()[i]  # 通过索引顺序获取
        for j in range(int(table.nrows)):
            if j == 0:
                index = index + 1
                label = str(0)
                name = "background"
                displayname = "背景"
            else:
                label = str(int(float(table.cell(j, 0).value)))
                displayname =table.cell(j,1).value
                name = table.cell(j,2).value
            with open(GetPreviousDir(xlsx_path)+"/"+protxt_name +".prototxt",'a') as f:
                context = "item{"+"\n\t"+"name: "+'"'+name+'"'+"\n\t"+"label: "+str(label)+"\n\t"+"display_name: "+'"'+displayname+'"'+"\n"+"}"+"\n"
                f.write(context)
        if i > 1:
            break

    print("GeneratePrototxt")


#xml准换成prototxt
def GeneratePbtxt(xlsx_path):
    data = xlrd.open_workbook(xlsx_path)
    table_count = len(data.sheets())
    for i in range(table_count):
        table = data.sheets()[i]  # 通过索引顺序获取
        for j in range(int(table.nrows)):
            if j == 0:
                continue
            else:
                label = str(int(float(table.cell(j, 0).value)))
                displayname =table.cell(j, 2).value
                name = table.cell(j,6).value
            with open("/home/jade/Data/StaticDeepFreeze/prototxt/class_"+str(i)+".pbtxt",'a') as f:
                context = "item {\n  "+"id: "+label+"\n  "+"name: "+"'"+name+"'"+"\n"+"}"+"\n"
                f.write(context)
    print("GeneratePbtxt")

def GetVOC_CLASSES(prototxt_path):
    categories,classes = ReadProTxt(prototxt_path)
    return classes

def GetVOC_Labels(prototxt_path):
    with open(prototxt_path,'r') as f:
        result = f.read().split("\n")
    protxt = []
    dics = {}
    class_names = []
    for i in range(len(result)):
        if i % 5 == 1:
            name = result[i].split('name: ')[1].split('"')[1]
            class_names.append(name)
        elif i % 5 == 2:
            label = int(result[i].split("label: ")[1])
        elif i % 5 == 3:
            display_name = result[i].split("display_name: ")[1].split('"')[1]
        elif i % 5 == 0 and i >0:
            if i ==5:
                dics.update({"none": (label,"Background")})
            else:
                dics.update({name: (label,name)})
    return dics

# VOC_LABELS = {
#     'none': (0, 'Background'),
#     'thumb_up': (1, 'thumb_up'),
#     'others':(2,'others')
#
# }


def model_string_sort(liststring):
    index_sort = [int(x.split(".")[1].split('-')[1])for x in liststring]
    index_sort.sort()
    max_index = index_sort[-1]
    return max_index

def GetModelPath(model_path):
    file_list = os.listdir(model_path)
    model_list = []
    for file in file_list:
        if 'model' in (file):
            model_list.append(file)
    max_index = model_string_sort(model_list)
    last_model_name = (model_list[-1])
    model_path_name = last_model_name.split(".")[0] + "." + last_model_name.split(".")[1].split("-")[0] + "-" + str(
        max_index)
    with open(os.path.join(model_path, "checkpoint"), 'w') as f:
        f.write('model_checkpoint_path: "' + os.path.join(model_path, model_path_name) + "\n")
    return os.path.join(model_path, model_path_name)



def GetModelStep(train_dir):
    file_list = os.listdir(train_dir)
    model_list = []
    for file in file_list:
        if 'model' in (file):
            model_list.append(file)
    if len(model_list) > 0 :
        max_index = model_string_sort(model_list)
        last_model_name = (model_list[-1])
        model_path_name = last_model_name.split(".")[0] + "." + last_model_name.split(".")[1].split("-")[0] + "-" + str(
            max_index)
        return max_index
    return 0


def ResizeClassifyDataset(classifyPath,size=224):
    filenames = os.listdir(classifyPath)
    processBar = ProcessBar()
    processBar.count = len(filenames)
    for filename in filenames:
        processBar.start_time = time.time()
        imagepaths = GetAllImagesPath(os.path.join(classifyPath, filename))
        for imagepath in imagepaths:
            image = cv2.imread(imagepath)
            image = cv2.resize(image,(size,size))
            newClassifyPath = CreateSavePath(classifyPath+"_resize")
            filepath = CreateSavePath(os.path.join(newClassifyPath,filename))
            cv2.imwrite(os.path.join(newClassifyPath,filename,GetLastDir(imagepath)),image)
        NoLinePrint("writing images..",processBar)

def VideoToImages(video_path,save_img_path):
    capture = cv2.VideoCapture(video_path)
    ret,frame = capture.read()
    index = 0
    while ret:
        if index % 5 == 0:
            cv2.imwrite(os.path.join(save_img_path,str(uuid.uuid1()))+".jpg",frame)
        print("*******正在写入中***************")
        index = index + 1
        ret,frame = capture.read()

def get_sys_path():
    for sys_path in sys.path:
        if "site-packages" in sys_path or 'dist-packages' in sys_path and os.path.isdir(sys_path):
            return sys_path
def get_anaconda_envs():
    sys_path = get_sys_path()
    envs_path = sys_path.split("envs")[0] + "envs"
    envs_list = os.listdir(envs_path)
    return envs_list

def get_envs_packages(python_path):
    if "python2.7" in (os.listdir(python_path+"/lib/")):
        return python_path + "/lib/python2.7/site-packages/"
    if "python3.6" in (os.listdir(python_path+"/lib/")):
        return python_path + "/lib/python3.6/site-packages/"

def get_anaconda_envs_path():
    sys_path = get_sys_path()
    envs_path = sys_path.split("envs")[0] + "envs"
    envs_list = os.listdir(envs_path)
    envs_path_list = []
    for envs_name in envs_list:
        envs_path_list.append(get_envs_packages(os.path.join(envs_path,envs_name)))
    envs_path_list.append(get_envs_packages(envs_path.split("anaconda3")[0]+"anaconda3"))
    return envs_path_list

def get_python_version():
    if "python3.6" in get_sys_path():
        return "python3"
    if "python2.7" in get_sys_path():
        return "python2.7"

if __name__ == '__main__':

    print(get_sys_path())
























