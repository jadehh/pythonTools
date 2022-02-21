#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : jade_opencv_process.py
# @Author   : jade
# @Date     : 2021/11/30 10:00
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
##旋转图片
import cv2
import numpy as np
from jade import ProgressBar,getOperationSystem
import threading
import random
import os
import time
import uuid
from opencv_tools import DIRECTORY_IMAGES,DIRECTORY_ANNOTATIONS,DIRECTORY_PREANNOTATIONS
import base64


## opencv读取中文路径图片
def imread_chinese_path(image_path):
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    return image

## 图像压缩
def Image_Resize(image,width=768):
    return cv2.resize(image,(width,int(width/image.shape[1]*image.shape[0])))
##旋转图片
def Image_Roate(image, angle):
    # 获取图像的尺寸
    # 旋转中心
    (h, w) = image.shape[:2]
    (cx, cy) = (w / 2, h / 2)

    # 设置旋转矩阵
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算图像旋转后的新边界
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy

    return cv2.warpAffine(image, M, (nW, nH))


def Video_Roate(video_path, save_video_path, angle, fps=20):
    video_capture = cv2.VideoCapture(video_path)
    ret, frame = video_capture.read()
    roate_img = Image_Roate(frame, angle)
    height = roate_img.shape[0]
    width = roate_img.shape[1]
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter(save_video_path, fourcc, fps, (width, height))
    progressBar = ProgressBar(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        frame = Image_Roate(frame, angle)
        videoWriter.write(frame)
        ret, frame = video_capture.read()
        progressBar.update()
        if ret is not True:
            break


# 分割视频
def split_video(input_video_path, output_video_path, start_time, end_time):
    """

    :param input_video_path: 输入视频地址
    :param output_video_path: 输出视频地址
    :param start_time: 开始时间
    :param end_time: 结束时间
    :return:
    """
    video_capture = cv2.VideoCapture(input_video_path)
    ret, frame = video_capture.read()
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    height = frame.shape[0]
    width = frame.shape[1]
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    if int(fps) == 0:
        fps = 15
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    index = 0
    while True:
        ret, frame = video_capture.read()
        index = index + 1
        current_time = index / fps
        if current_time > start_time and current_time < end_time:
            video_writer.write(frame)
            print("Writing Videos")
        if current_time > end_time:
            break
        if ret is False:
            break


class VideoCapture():
    def __init__(self, cv_videocapture_param):
        self.cv_videocapture_param = cv_videocapture_param
        return

    def handle(self):
        # 开启线程，传入参数
        _thread = threading.Thread(target=self.open)
        _thread.setDaemon(True)
        _thread.start()  # 启动线程
        return

    def open(self):
        capture = cv2.VideoCapture(self.cv_videocapture_param)
        ret, frame = capture.read()
        while ret:
            print("正在开启线程读取视频")
            ret, frame = capture.read()








class processImage:
    def __init__(self, img):
        self.img = img

    # RGB转BGR
    def RGBTOBGR(self):
        return cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)

    # BGR转RGB，一般用于图像不能显示正常的颜色
    def BGRTORGB(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

    # 给图像画一个矩形框,默认为随机的颜色,bboxes为xmin/width
    def RECTANGLE(self, bboxes):
        shape = self.img.shape
        width = shape[1]
        height = shape[0]
        color_R = random.randint(1, 254)
        color_G = random.randint(1, 254)
        color_B = random.randint(1, 255)
        self.img = cv2.rectangle(self.img, (int(bboxes[1] * width), int(bboxes[0] * height)),
                                 (int(bboxes[2] * width), int(bboxes[3] * height)), (color_R, color_G, color_B), 2, 2)
        return self.img

    # 图像画点
    def CIRCLE(self, points):
        color_R = random.randint(1, 254)
        color_G = random.randint(1, 254)
        color_B = random.randint(1, 255)
        for point in points:
            self.img = cv2.circle(self.img, point, 2, (color_R, color_G, color_B), 2, 2)
        return self.img

    # 高斯去噪
    def Gaussian_Blur(self):
        blurred = cv2.GaussianBlur(self.img, (9, 9), 0)
        return blurred

    # 高斯去噪后阈值分割,返回彩色图像
    def Thresh_and_Blur(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(img, (9, 9), 0)
        (thresh_value, thresh) = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)
        for c in range(3):
            self.img[:, :, c] = np.where(thresh[:, :, ] == 0,
                                         0,
                                         self.img[:, :, c])
        return self.img

    # otsu阈值分割,返回彩色图
    def ThreshColor(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        (thresh_value, thresh) = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        for c in range(3):
            self.img[:, :, c] = np.where(thresh[:, :, ] == 0,
                                         0,
                                         self.img[:, :, c])
        return self.img

    # ousu阈值分割，灰度图
    def ThreshGray(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        (thresh_value, thresh) = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        return thresh

    # 图像倾斜矫正,一般是图像分割后做倾斜矫正
    def image_rectification(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh == 0))
        angle = cv2.minAreaRect(coords)[-1]
        # print(angle)
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        h = self.img.shape[0]
        w = self.img.shape[1]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(self.img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        target_coords = np.where(rotated[:, :, 1] > 0)
        if len(target_coords[0]) > 0:
            rotated = rotated[min(target_coords[0]):max(target_coords[0]), min(target_coords[1]):max(target_coords[1]),
                      :]
        # 需要裁剪边框
        return rotated






# opencv 读取中文
def ReadChinesePath(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    return cv_img

def WriteChienePath(file_path,image):
    cv2.imencode('.jpg', image)[1].tofile(file_path)

# 随机出一个颜色
def GetRandomColor():
    r1 = random.randint(0, 255)
    r2 = random.randint(0, 255)
    r3 = random.randint(0, 255)
    return (r1, r2, r3)


def opencv_to_base64(image):
    image_byte = cv2.imencode('.jpg', image)[1].tobytes()
    base64_str = str(base64.b64encode(image_byte), encoding='utf-8')
    return base64_str





# 图像裁剪边框
def ImageRectification(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh == 0))
    angle = cv2.minAreaRect(coords)[-1]
    # print(angle)
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    h = img.shape[0]
    w = img.shape[1]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    target_coords = np.where(rotated[:, :, 1] > 0)
    rotated = rotated[min(target_coords[0]):max(target_coords[0]), min(target_coords[1]):max(target_coords[1]), :]
    # 需要裁剪边框
    return rotated


# PLT显示图片关键点和矩形框
def PltShowKeypointsBoxes(img_path, keypoints, bboxes=[], scores=[], waitkey=1):
    if type(img_path) == str:
        im = plt.imread(img_path)
    else:
        im = img_path
    plt.axis("off")
    plt.imshow(im)
    pts = np.array(keypoints)
    scores = np.array(scores)
    for i in range(len(pts)):
        score = (scores[i]).mean()
        pt = pts[i]
        if score > 0.5:
            currentAxis = plt.gca()
            rect = patches.Rectangle((bboxes[i][0], bboxes[i][1]), bboxes[i][2] - bboxes[i][0],
                                     bboxes[i][3] - bboxes[i][1],
                                     linewidth=1,
                                     edgecolor='r', facecolor='none')
            currentAxis.add_patch(rect)
            for p in range(pt.shape[0]):
                score2 = scores[i][p, 0]
                if score2 > 0.5 and p in [5, 6, 7, 8, 9, 10]:
                    plt.plot(pt[p, 0], pt[p, 1], 'r.')
                    plt.text(pt[p, 0], pt[p, 1], '{0}'.format(p))
    edges = [[5, 7], [7, 9], [6, 8], [8, 10]]
    for i in range(len(pts)):
        for ie, e in enumerate(edges):
            rgb = matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
            plt.plot(pts[i][e, 0], pts[i][e, 1], color=rgb)
    plt.ion()
    plt.pause(waitkey)  # 显示的时间
    plt.close()


def CVShowKeyPoints(image, keyPoints, classes=None, waiktKey=1, named_windows="result"):
    base = int(np.ceil(pow(len(keyPoints), 1. / 3)))
    colors = [_to_color(x) for x in range(len(keyPoints))]
    h, w = image.shape[0], image.shape[1]
    for i in range(len(keyPoints)):
        for j in range(len(keyPoints[i])):
            if keyPoints[i][j][0] < 0:
                image = cv2.circle(image, (int(keyPoints[i][j][0] * w), int(keyPoints[i][j][1] * h)), 1, colors[i], 3,
                                   3)
                # image = Add_Chinese_Label(image, "{}".format(j), (int(keyPoints[i][j][0]*w),int(keyPoints[i][j][1]*h)), colors[i], 24)
            else:
                image = cv2.circle(image, (int(keyPoints[i][j][0]), int(keyPoints[i][j][1])), 1, colors[i], 3,
                                   3)
                # image = Add_Chinese_Label(image, "{}".format(j), (int(keyPoints[i][j][0]),int(keyPoints[i][j][1])), colors[i], 24)

        point1 = (int(keyPoints[i][0][0]), int(keyPoints[i][0][1]))
        point2 = (int(keyPoints[i][1][0]), int(keyPoints[i][1][1]))
        point3 = (int(keyPoints[i][2][0]), int(keyPoints[i][2][1]))
        point4 = (int(keyPoints[i][3][0]), int(keyPoints[i][3][1]))
        image = cv2.line(image, point1, point2, colors[i], 2, 2)
        image = cv2.line(image, point2, point3, colors[i], 2, 2)
        image = cv2.line(image, point3, point4, colors[i], 2, 2)
        image = cv2.line(image, point4, point1, colors[i], 2, 2)
        if classes:
            image = Add_Chinese_Label(image, classes[i], point1, colors[i], 40)

    if waiktKey >= 0:
        cv2.namedWindow(named_windows, 0)
        cv2.imshow(named_windows, image)
        cv2.waitKey(waiktKey)
    else:
        return image

        return image


# opencv 转 base64
def cv2_base64(image):
    base64_str = cv2.imencode('.jpg', image)[1].tostring()
    base64_str = base64.b64encode(base64_str)
    return str(base64_str, encoding="utf-8")


# opencv显示关键点和矩形框
def CVShowKeypointsBoxes(img_path, keypoints, bboxes=[], scores=[], waitkey=1):
    if type(img_path) == str:
        im = plt.imread(img_path)
    else:
        im = img_path

    pts = np.array(keypoints)
    scores = np.array(scores)
    edges = [[5, 7], [7, 9], [6, 8], [8, 10]]
    for i in range(len(pts)):
        score = (scores[i]).mean()
        pt = pts[i]
        if score > 0.5:
            im = cv2.rectangle(im, (int(bboxes[i][0]), int(bboxes[i][1])), (int(bboxes[i][2]), int(bboxes[i][3])),
                               (255, 255, 255), 2, 2)
            for p in range(pt.shape[0]):
                score2 = scores[i][p, 0]
                # if score2 > 0.5 and p in [5,6,7,8,9,10]:
                im = cv2.circle(im, (int(pt[p, 0]), int(pt[p, 1])), 1, (255, 0, 0), 3, 3)
                im = cv2.putText(im, str(p), (int(pt[p, 0]), int(pt[p, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                 (0, 0, 0), 1)

            for ie, e in enumerate(edges):
                rgb = matplotlib.colors.hsv_to_rgb([ie / float((len(edges))), 1, 1])

                im = cv2.line(im, (int(pt[e[0]][0]), int(pt[e[0]][1])),
                              (int(pt[e[1]][0]), int(pt[e[1]][1])), rgb * 255, 3, 3)

    cv2.namedWindow("result", 0)
    cv2.resizeWindow("result", 840, 680)
    cv2.imshow("result", im)
    cv2.waitKey(waitkey)


def _to_color(indx):
    """ return (b, r, g) tuple"""
    b = random.randint(1, 10) / 10
    g = random.randint(1, 10) / 10
    r = random.randint(1, 10) / 10
    return b * 255, r * 255, g * 255


# opencv显示boxes
def CVShowBoxes(image, detectresult, num_classes=90, waitkey=-1, named_windows="result"):
    base = int(np.ceil(pow(num_classes, 1. / 3)))
    colors = [_to_color(x) for x in range(num_classes)]
    if type(image) == str:
        image = cv2.imread(image)
    image2 = image.copy()
    boxes = detectresult.boxes
    for i in range(len(boxes)):
        if boxes[i][0] <= 1 and boxes[i][1] <= 1 and boxes[i][2] <= 1 and boxes[i][3] <= 1:
            xmin = int(boxes[i][0] * image.shape[1])
            ymin = int(boxes[i][1] * image.shape[0])
            xmax = int(boxes[i][2] * image.shape[1])
            ymax = int(boxes[i][3] * image.shape[0])
        else:
            xmin = int(boxes[i][0])
            ymin = int(boxes[i][1])
            xmax = int(boxes[i][2])
            ymax = int(boxes[i][3])
        if boxes is not None:
            image2 = cv2.rectangle(image2, (xmin, ymin), (xmax, ymax), GetRandomColor(), 3, 3)
            if detectresult.label_texts is not None:
                if detectresult.scores is not None:
                    image2 = Add_Chinese_Label(img=image2, label=detectresult.label_texts[i] + ":" + str(
                        int(detectresult.scores[i] * 100)),
                                               pt1=(xmin, ymin))
                else:
                    image2 = Add_Chinese_Label(img=image2, label=detectresult.label_texts[i],
                                               pt1=(xmin, ymin))
                if detectresult.label_ids is not None:
                    image2 = cv2.rectangle(image2, (xmin, ymin), (xmax, ymax), colors[int(detectresult.label_ids[i])],
                                           3, 3)
                else:
                    image2 = cv2.rectangle(image2, (xmin, ymin), (xmax, ymax), GetRandomColor(), 3, 3)

    if waitkey >= 0:
        cv2.namedWindow(named_windows, 0)
        # cv2.resizeWindow("result", 840, 680)
        cv2.imshow(named_windows, image2)
        cv2.waitKey(waitkey)
    else:
        return image2


# opencv显示points
def CVShowPoints(img_path, points, waitkey=1):
    if type(img_path) != list:
        image = cv2.imread(img_path)
    else:
        image = img_path
    image2 = image.copy()

    for i in range(len(points)):
        psts = []
        points2 = points[i]
        for j in range(len(points2)):
            for z in range(len(points2[j])):
                if z % 2 == 0 and z != 0:
                    cv2.circle(image2, (int(points2[j][z - 1]), int(points2[j][z])), 2, (255, 255, 255), 2, 1)

    cv2.imshow("resukt", image2)
    cv2.waitKey(waitkey * 1000)


# PLT显示关键点
def PltShowKeypoints(img_path, keypoints, waitkey=1):
    if type(img_path) != list:
        im = plt.imread(img_path)
    else:
        im = img_path
    edges = [[0], [1, 3], [3, 5], [2, 4], [4, 6]]
    plt.imshow(im)
    plt.axis("off")
    pts = np.array(keypoints)
    for i in range(len(pts)):
        points2 = pts[i]
        for j in range(len(points2)):
            for z in range(len(points2[j])):
                if z % 2 == 0 and z != 0:
                    plt.plot(int(points2[j][z - 1]), int(points2[j][z]), 'r.')
                    plt.text(int(points2[j][z - 1]), int(points2[j][z]), '{0}'.format(j))

        # for ie, e in enumerate(edges):
        #     rgb = matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
        #     if len(points2) == 7:
        #         plt.plot(points2[e, 1], points2[e, 2], color=rgb)
    plt.ion()
    plt.pause(waitkey)  # 显示的时间
    plt.close()


# 合并图片
def CombinedImages(images, img_per_row=3, columns=3):
    inputs = []
    for img in images:
        # img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        inputs.append(Image.fromarray(img.astype('uint8')).convert('RGB'))

    width, height = inputs[0].size
    img = Image.new(inputs[0].mode, (width * img_per_row, height * columns))
    idx = 0
    for row in range(img_per_row):
        for col in range(columns):
            if idx > len(images) - 1:
                break
            img.paste(inputs[idx], box=(row * width, col * height))
            idx = idx + 1

    img = np.array(img)
    return img


# 裁剪目标框
def CutImageWithBox(image, bbox,expand_size=20):
    xmin = (int(bbox[0] - expand_size), 0)[int(bbox[0] - expand_size) < 0]
    ymin = (int(bbox[1] - expand_size), 0)[int(bbox[1] - expand_size) < 0]
    xmax = (int(bbox[2] + expand_size), 0)[int(bbox[2] + expand_size) < 0]
    ymax = (int(bbox[3] + expand_size), 0)[int(bbox[3] + expand_size) < 0]
    cut_image = image[ymin:ymax, xmin:xmax, :]
    return cut_image


def CutImageWithBoxes(image, bboxes, convert_rgb2bgr=False):
    cut_images = []
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        cut_image = CutImageWithBox(image, bbox)
        cut_images.append(cut_image)
        # cut_images.append(cv2.resize(cut_image,(256,256)))

    return cut_images


def GetLabelAndImagePath(root_path):
    image_labels = {}
    labels = os.listdir(root_path)
    for label in labels:
        image_paths = GetAllImagesPath(os.path.join(root_path, label))
        for image_path in image_paths:
            image_labels.update({GetLastDir(image_path)[:-4]: label})
    return image_labels


# VOC数据集裁剪目标框
def CutImagesWithVoc(xml_path):
    imagename, shape, bboxes, labels, labels_text, difficult, truncated = ProcessXml(xml_path)
    root_path = GetPreviousDir(GetPreviousDir(xml_path))
    image_path = os.path.join(root_path, DIRECTORY_IMAGES, GetLastDir(xml_path)[:-4] + ".jpg")
    image = cv2.imread(image_path)
    images = CutImageWithBoxes(image, bboxes)
    return images, labels


# 读取自定义标注的MASK
def LoadEraseMask(image_path):
    if type(image_path) == str:
        img = cv2.imread(image_path)
    else:
        img = image_path
    img = cv2.resize(img, (JADE_RESIZE_SIZE, JADE_RESIZE_SIZE))
    input_height = img.shape[1]
    input_width = img.shape[0]

    # mouse callback function
    def erase_rect(event, x, y, flags, param):
        global ix, iy, JADE_DRAWING
        if event == cv2.EVENT_LBUTTONDOWN:
            JADE_DRAWING = True
            if JADE_DRAWING == True:
                # cv2.circle(img,(x,y),10,(255,255,255),-1)
                cv2.rectangle(img, (x - JADE_SIZE, y - JADE_SIZE), (x + JADE_SIZE, y + JADE_SIZE), JADE_COLOR, -1)
                cv2.rectangle(mask, (x - JADE_SIZE, y - JADE_SIZE), (x + JADE_SIZE, y + JADE_SIZE), JADE_COLOR, -1)

        elif event == cv2.EVENT_MOUSEMOVE:
            if JADE_DRAWING == True:
                # cv2.circle(img,(x,y),10,(255,255,255),-1)
                cv2.rectangle(img, (x - JADE_SIZE, y - JADE_SIZE), (x + JADE_SIZE, y + JADE_SIZE), JADE_COLOR, -1)
                cv2.rectangle(mask, (x - JADE_SIZE, y - JADE_SIZE), (x + JADE_SIZE, y + JADE_SIZE), JADE_COLOR, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            JADE_DRAWING = False
            # cv2.circle(img,(x,y),10,(255,255,255),-1)
            cv2.rectangle(img, (x - JADE_SIZE, y - JADE_SIZE), (x + JADE_SIZE, y + JADE_SIZE), JADE_COLOR, -1)
            cv2.rectangle(mask, (x - JADE_SIZE, y - JADE_SIZE), (x + JADE_SIZE, y + JADE_SIZE), JADE_COLOR, -1)

    mask = np.zeros(img.shape)
    test_mask = cv2.resize(mask, (input_height, input_width))
    test_mask = test_mask.astype(np.uint8)
    test_mask = cv2.cvtColor(test_mask, cv2.COLOR_RGB2GRAY)
    cv2.destroyAllWindows()

    cv2.namedWindow('image', 0)
    cv2.setMouseCallback('image', erase_rect)
    # cv2.namedWindow('mask')
    #
    mask = np.zeros(img.shape, dtype=np.uint8)

    while (1):
        img_show = img
        cv2.imshow('image', img_show)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    test_img = cv2.resize(img, (input_height, input_width))
    test_mask = cv2.resize(mask, (input_height, input_width))
    test_mask = cv2.cvtColor(test_mask, cv2.COLOR_RGB2GRAY)
    return test_mask


# 从mask文件夹随机读取mask文件
def LoadRandomMask(mask_count=100):
    mask_path = "/home/jade/Data/mask/testing_mask_dataset"
    mask_list = GetAllImagesPath(mask_path)
    mask_list = mask_list[5000:10000]
    masks = random.sample(mask_list, mask_count)
    return masks


# 在图片中添加Mask区域
def MixImageAndMask(image, mask):
    if type(mask) == str:
        mask = cv2.imread(mask)
    image = cv2.resize(image, (512, 512))
    mask = mask > 0
    image[mask] = 255
    return image


# 旋转图片
def RotateBound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


# 鱼眼矫正
def get_K_and_D(checkerboard, imgsPath):
    CHECKERBOARD = checkerboard
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    _img_shape = None
    objpoints = []
    imgpoints = []
    images = glob.glob(imgsPath + '/*.jpg')
    for fname in images:
        print(fname)
        img = cv2.imread(fname)
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
        # cv2.namedWindow("result",0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("result",gray)
        # cv2.waitKey(0)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
            imgpoints.append(corners)
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

    rms, _, _, _, _ = \
        cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    DIM = _img_shape[::-1]
    return DIM, K, D


# 裁剪矫正
def undistort1(img_path, DIM=(1920, 1080), K=np.array([[9.10274325e+02, 0.00000000e+00, 1.03283696e+03],
                                                       [0.00000000e+00, 9.13958936e+02, 5.80558859e+02],
                                                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
               D=np.array([[-0.06510345], [-0.01617996], [0.18211916], [-0.15394744]])):
    img = cv2.imread(img_path)
    img = cv2.resize(img, DIM)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img
    # cv2.destroyAllWindows()


# 无裁剪矫正
def undistort2(img_path, DIM=(1920, 1080), K=np.array([[9.10274325e+02, 0.00000000e+00, 1.03283696e+03],
                                                       [0.00000000e+00, 9.13958936e+02, 5.80558859e+02],
                                                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
               D=np.array([[-0.06510345], [-0.01617996], [0.18211916], [-0.15394744]]), balance=0.6, dim2=None,
               dim3=None):
    img = cv2.imread(img_path)
    dim1 = img.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
    assert dim1[0] / dim1[1] == DIM[0] / DIM[
        1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img



# 坐标旋转
def BoxRotated(image_path, bboxes):
    image1 = cv2.imread(image_path)
    height, width, C = image1.shape
    new_bboxes = []
    for i in range(len(bboxes)):
        bbox2 = [width - bboxes[i][0], height - bboxes[i][1], width - bboxes[i][2], height - bboxes[i][3]]
        new_bboxes.append(bbox2)
    return new_bboxes


def ImShow(images, key=0):
    if type(images) == list:
        for i in range(len(images)):
            if type(images[i]) == str:
                image = cv2.imread(images[i])
            else:
                image = images[i]
            name = "result_" + str(i)
            cv2.namedWindow(name, 0)
            cv2.imshow(name, image)
    else:
        if type(images) == str:
            image = cv2.imread(images)
        else:
            image = images
        name = "result_0"
        cv2.namedWindow(name, 0)
        cv2.imshow(name, image)
    cv2.waitKey(key)


def compose_gif(image_path_list, output_path, fps=1):
    import imageio
    gif_images = []
    for path in image_path_list:
        gif_images.append(imageio.imread(path))
    imageio.mimsave(output_path, gif_images, fps=fps)


def overlay_image(image_path1,image_path2,boxes):
    background_image = ReadChinesePath(image_path1)
    image = ReadChinesePath(image_path2)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    width = boxes[3]
    height = boxes[2]
    background_image[boxes[0]:boxes[0]+boxes[2],boxes[1]:boxes[1]+boxes[3],:] = cv2.resize(image,(width,height))
    cv2.namedWindow("result",0)
    cv2.imshow("result",background_image)
    cv2.waitKey(0)
    return background_image

if __name__ == '__main__':
    image = ReadChinesePath(r"F:\现场数据\镇江大港\车牌图片\2021-12-15\20211215150146781771.jpg")
    image = Image_Resize(image,768)
    overlay_image(r"F:\现场数据\镇江大港\车牌图片\2021-12-15\20211215150146781771.jpg",r"C:\Users\Administrator\Desktop\联图二维码.png",[880,1380,150,150])