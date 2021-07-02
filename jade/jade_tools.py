#coding=utf-8
import datetime
import sys
import time
import cv2
import os
import xlrd
import os.path as ops
from jade import *

import sys
from collections.abc import Iterable
from multiprocessing import Pool
from shutil import get_terminal_size


class TimerError(Exception):

    def __init__(self, message):
        self.message = message
        super(TimerError, self).__init__(message)


class Timer:
    """A flexible Timer class.

    :Example:

    >>> import time
    >>> import mmcv
    >>> with mmcv.Timer():
    >>>     # simulate a code block that will run for 1s
    >>>     time.sleep(1)
    1.000
    >>> with mmcv.Timer(print_tmpl='it takes {:.1f} seconds'):
    >>>     # simulate a code block that will run for 1s
    >>>     time.sleep(1)
    it takes 1.0 seconds
    >>> timer = mmcv.Timer()
    >>> time.sleep(0.5)
    >>> print(timer.since_start())
    0.500
    >>> time.sleep(0.5)
    >>> print(timer.since_last_check())
    0.500
    >>> print(timer.since_start())
    1.000
    """

    def __init__(self, start=True, print_tmpl=None):
        self._is_running = False
        self.print_tmpl = print_tmpl if print_tmpl else '{:.3f}'
        if start:
            self.start()

    @property
    def is_running(self):
        """bool: indicate whether the timer is running"""
        return self._is_running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        print(self.print_tmpl.format(self.since_last_check()))
        self._is_running = False

    def start(self):
        """Start the timer."""
        if not self._is_running:
            self._t_start = time.time()
            self._is_running = True
        self._t_last = time.time()

    def since_start(self):
        """Total time since the timer is started.

        Returns (float): Time in seconds.
        """
        if not self._is_running:
            raise TimerError('timer is not running')
        self._t_last = time.time()
        return self._t_last - self._t_start

    def since_last_check(self):
        """Time since the last checking.

        Either :func:`since_start` or :func:`since_last_check` is a checking
        operation.

        Returns (float): Time in seconds.
        """
        if not self._is_running:
            raise TimerError('timer is not running')
        dur = time.time() - self._t_last
        self._t_last = time.time()
        return dur


_g_timers = {}  # global timers


def check_time(timer_id):
    """Add check points in a single line.

    This method is suitable for running a task on a list of items. A timer will
    be registered when the method is called for the first time.

    :Example:

    >>> import time
    >>> import mmcv
    >>> for i in range(1, 6):
    >>>     # simulate a code block
    >>>     time.sleep(i)
    >>>     mmcv.check_time('task1')
    2.000
    3.000
    4.000
    5.000

    Args:
        timer_id (str): Timer identifier.
    """
    if timer_id not in _g_timers:
        _g_timers[timer_id] = Timer()
        return 0
    else:
        return _g_timers[timer_id].since_last_check()



class ProgressBar:
    """A progress bar which can print the progress."""

    def __init__(self, task_num=0, bar_width=50, start=True, file=sys.stdout):
        self.task_num = task_num
        self.bar_width = bar_width
        self.completed = 0
        self.file = file
        if start:
            self.start()

    @property
    def terminal_width(self):
        width, _ = get_terminal_size()
        return width

    def start(self):
        if self.task_num > 0:
            self.file.write(f'[{" " * self.bar_width}] 0/{self.task_num}, '
                            '花费了: 0s, 预计还剩:')
        else:
            self.file.write('完成: 0, 共花费: 0s')
        self.file.flush()
        self.timer = Timer()

    def update(self, num_tasks=1):
        assert num_tasks > 0
        self.completed += num_tasks
        elapsed = self.timer.since_start()
        if elapsed > 0:
            fps = self.completed / elapsed
        else:
            fps = float('inf')
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            msg = f'\r[{{}}] {self.completed}/{self.task_num}, ' \
                  f'{fps:.1f} task/s, 花费了: {int(elapsed + 0.5)}s, ' \
                  f'预计还剩: {eta:5}s'

            bar_width = min(self.bar_width,
                            int(self.terminal_width - len(msg)) + 2,
                            int(self.terminal_width * 0.6))
            bar_width = max(2, bar_width)
            mark_width = int(bar_width * percentage)
            bar_chars = '>' * mark_width + ' ' * (bar_width - mark_width)
            self.file.write(msg.format(bar_chars))
        else:
            self.file.write(
                f'完成: {self.completed}, 共花费: {int(elapsed + 0.5)}s,'
                f' {fps:.1f} tasks/s')
        self.file.flush()


def track_progress(func, tasks, bar_width=50, file=sys.stdout, **kwargs):
    """Track the progress of tasks execution with a progress bar.

    Tasks are done with a simple for-loop.

    Args:
        func (callable): The function to be applied to each task.
        tasks (list or tuple[Iterable, int]): A list of tasks or
            (tasks, total num).
        bar_width (int): Width of progress bar.

    Returns:
        list: The task results.
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, Iterable):
        task_num = len(tasks)
    else:
        raise TypeError(
            '"tasks" must be an iterable object or a (iterator, int) tuple')
    prog_bar = ProgressBar(task_num, bar_width, file=file)
    results = []
    for task in tasks:
        results.append(func(task, **kwargs))
        prog_bar.update()
    prog_bar.file.write('\n')
    return results


def init_pool(process_num, initializer=None, initargs=None):
    if initializer is None:
        return Pool(process_num)
    elif initargs is None:
        return Pool(process_num, initializer)
    else:
        if not isinstance(initargs, tuple):
            raise TypeError('"initargs" must be a tuple')
        return Pool(process_num, initializer, initargs)


def track_parallel_progress(func,
                            tasks,
                            nproc,
                            initializer=None,
                            initargs=None,
                            bar_width=50,
                            chunksize=1,
                            skip_first=False,
                            keep_order=True,
                            file=sys.stdout):
    """Track the progress of parallel task execution with a progress bar.

    The built-in :mod:`multiprocessing` module is used for process pools and
    tasks are done with :func:`Pool.map` or :func:`Pool.imap_unordered`.

    Args:
        func (callable): The function to be applied to each task.
        tasks (list or tuple[Iterable, int]): A list of tasks or
            (tasks, total num).
        nproc (int): Process (worker) number.
        initializer (None or callable): Refer to :class:`multiprocessing.Pool`
            for details.
        initargs (None or tuple): Refer to :class:`multiprocessing.Pool` for
            details.
        chunksize (int): Refer to :class:`multiprocessing.Pool` for details.
        bar_width (int): Width of progress bar.
        skip_first (bool): Whether to skip the first sample for each worker
            when estimating fps, since the initialization step may takes
            longer.
        keep_order (bool): If True, :func:`Pool.imap` is used, otherwise
            :func:`Pool.imap_unordered` is used.

    Returns:
        list: The task results.
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, Iterable):
        task_num = len(tasks)
    else:
        raise TypeError(
            '"tasks" must be an iterable object or a (iterator, int) tuple')
    pool = init_pool(nproc, initializer, initargs)
    start = not skip_first
    task_num -= nproc * chunksize * int(skip_first)
    prog_bar = ProgressBar(task_num, bar_width, start, file=file)
    results = []
    if keep_order:
        gen = pool.imap(func, tasks, chunksize)
    else:
        gen = pool.imap_unordered(func, tasks, chunksize)
    for result in gen:
        results.append(result)
        if skip_first:
            if len(results) < nproc * chunksize:
                continue
            elif len(results) == nproc * chunksize:
                prog_bar.start()
                continue
        prog_bar.update()
    prog_bar.file.write('\n')
    pool.close()
    pool.join()
    return results


def track_iter_progress(tasks, bar_width=50, file=sys.stdout):
    """Track the progress of tasks iteration or enumeration with a progress
    bar.

    Tasks are yielded with a simple for-loop.

    Args:
        tasks (list or tuple[Iterable, int]): A list of tasks or
            (tasks, total num).
        bar_width (int): Width of progress bar.

    Yields:
        list: The task results.
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, Iterable):
        task_num = len(tasks)
    else:
        raise TypeError(
            '"tasks" must be an iterable object or a (iterator, int) tuple')
    prog_bar = ProgressBar(task_num, bar_width, file=file)
    for task in tasks:
        yield task
        prog_bar.update()
    prog_bar.file.write('\n')





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
        return "python2"

def clear_queue(queue):
    qsize = queue.qsize()
    for i in range(qsize):
        queue.get()


def get_Ip_address():
    ipaddress = os.popen("ifconfig",'r')
    ifconfig_list = []
    for line in ipaddress:
        ifconfig_list.append(line)
    wlan_list = []
    config_tmp = ""
    for config in ifconfig_list:
        if config == "\n":
            wlan_list.append(config_tmp)
            config_tmp = ""
        else:
            config_tmp = config_tmp + config.split("\n")[0]

    for wlan_config in wlan_list:
        if "255.255." in wlan_config and "docker" not in wlan_config:
            return wlan_config.split("inet")[1].split("netmask")[0].strip()




if __name__ == '__main__':
    import time
    progressBar  = ProgressBar(10)

    for i in range(10):
        time.sleep(1)
        progressBar.update()







