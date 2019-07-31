#coding=utf-8
from jade import *
import numpy as np
from xml.dom import minidom
import xml.etree.ElementTree as ET

def GetXmlClassesNames(xml_path):
    classnames = []
    # Read the XML annotation file.
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Image shape.
    size = root.find('size')
    roorname = root.find('filename').text
    shape = [(size.find('height').text),
             (size.find('width').text),
             (size.find('depth').text)]
    # Find annotations.
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    classnames = []
    for obj in root.findall('object'):
        #label = (obj.find('bndbox')).find('name').text
        label = obj.find('name').text
        if label not in classnames:
            classnames.append(label)
    return classnames







def ProcessXml(xml_path):
    # Read the XML annotation file.
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Image shape.
    size = root.find('size')
    roorname = root.find('filename').text
    shape = [(size.find('height').text),
             (size.find('width').text),
             (size.find('depth').text)]
    # Find annotations.
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        #label = (obj.find('bndbox')).find('name').text
        label = obj.find('name').text
        if label in VOC_LABELS.keys():
            labels.append(str(VOC_LABELS[label][0]))
        else:
            labels.append(1)
        labels_text.append(label)

        if obj.find('difficult'):
            difficult.append((obj.find('difficult').text))
        else:
            difficult.append('0')
        if obj.find('truncated'):
            truncated.append((obj.find('truncated').text))
        else:
            truncated.append('0')

        bbox = obj.find('bndbox')
        bboxes.append((int(bbox.find('xmin').text) / int(shape[1]) ,int(bbox.find('ymin').text) / int(shape[0]),int(bbox.find('xmax').text)/int(shape[1]),int(bbox.find('ymax').text)/int(shape[0])))
    imagename = GetLastDir(xml_path)[:-4]+'.jpg'
    return imagename,shape, bboxes, labels_text,labels, difficult, truncated




def ProcessXml_Dataset(xml_path):
    # Read the XML annotation file.
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Image shape.
    size = root.find('size')
    roorname = root.find('filename').text
    shape = [float((size.find('height').text)),
             float((size.find('width').text)),
             float((size.find('depth').text))]
    # Find annotations.
    ground_truth = []
    labels = []
    for obj in root.findall('object'):
        #label = (obj.find('bndbox')).find('name').text
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        ground_truth.append(np.array([int(bbox.find('xmin').text) / shape[1],
                                 int(bbox.find('ymin').text) / shape[0],
                                 int(bbox.find('xmax').text)/shape[1],
                                 int(bbox.find('ymax').text)/shape[0],int(VOC_LABELS[label][0])]))
    new_groudth = []
    for i in range(10):
        if  i < len(ground_truth):
            new_groudth.append(ground_truth[i])
        else:
            new_groudth.append([0,0,0,0,0])
    return  np.array(new_groudth)

# 生成XML文件方式
def GenerateXml(filename,shape,bboxes,labels_text,save_path):
    CreateSavePath(save_path)
    # 为根元素添加10个子元素
    difficult = [0]*len(bboxes)
    truncated = [0]*len(bboxes)
    obj_numbsers = len(labels_text)
    childElementname = ['folder','filename','source','size','segmented','object']
    sourcename = ['database','annotation','image']
    source_value = ['The Hand Database','VOC Hand','flickr']
    sizename = ['width','height','depth']
    width = shape[1]
    height = shape[0]
    depth = shape[2]
    size_value = [width,height,depth]
    objname = ['name','pose','difficult','truncated','bndbox']
    bndbox_key = ['xmin','ymin','xmax','ymax']
    obj_value = [labels_text,['Unspecified']*obj_numbsers,difficult,truncated]
    bndbox_value = bboxes
    doc = minidom.Document()
    annotation = doc.createElement("annotation")
    doc.appendChild(annotation)
    for i in range(6):

        if i==0:
            secname = doc.createElement(childElementname[i])
            annotation.appendChild(secname)
            source_name = doc.createTextNode("JPEGImages")
            secname.appendChild(source_name)
        if i==1:
            secname = doc.createElement(childElementname[i])
            annotation.appendChild(secname)
            source_name = doc.createTextNode(filename)
            secname.appendChild(source_name)
        if i == 2:
            secname = doc.createElement(childElementname[i])
            annotation.appendChild(secname)
            for j in range(len(sourcename)):
                thirdname = doc.createElement(sourcename[j])
                secname.appendChild(thirdname)
                source_name = doc.createTextNode(source_value[j])
                thirdname.appendChild(source_name)
        if i == 3:
            secname = doc.createElement(childElementname[i])
            annotation.appendChild(secname)
            for k in range(len(sizename)):
                thirdname = doc.createElement(sizename[k])
                secname.appendChild(thirdname)
                source_name = doc.createTextNode(str(size_value[k]))
                thirdname.appendChild(source_name)
        if i == 4:
            secname = doc.createElement(childElementname[i])
            annotation.appendChild(secname)
            source_name = doc.createTextNode("0")
            secname.appendChild(source_name)
        if i == 5:
            for n in range(obj_numbsers):
                secname = doc.createElement(childElementname[i])
                annotation.appendChild(secname)
                for k in range(len(objname)):
                    thirdname = doc.createElement(objname[k])
                    secname.appendChild(thirdname)
                    if k == 4:
                        for m in range(4):
                            forthname = doc.createElement(bndbox_key[m])
                            thirdname.appendChild(forthname)
                            if m==0:
                                xmin = int(bndbox_value[n][m])
                                if xmin < 0:
                                    xmin = 0
                                source_name = doc.createTextNode(str(xmin))
                            if m==1:
                                ymin = int(float(bndbox_value[n][m]))
                                if ymin < 0:
                                    ymin = 0
                                source_name = doc.createTextNode(str(ymin))
                            if m==2:
                                xmax = int(bndbox_value[n][m])
                                if xmax > width:
                                    xmax = width
                                source_name = doc.createTextNode(str(xmax))
                            if m==3:
                                ymax = int(bndbox_value[n][m])
                                if ymax > height:
                                    ymax = height
                                source_name = doc.createTextNode(str(ymax))
                            forthname.appendChild(source_name)

                    else:
                        source_name = doc.createTextNode(str(obj_value[k][n]))
                        thirdname.appendChild(source_name)

    save_xml_path = os.path.join(save_path,filename+".xml")
    f = open(save_xml_path, "w")
    f.write(doc.toprettyxml(indent="  "))
    f.close()


def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录 # 创建目录操作函数
        os.makedirs(path)

        print (path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print (path + ' 目录已存在')
        return False
#smb://192.168.1.202/data/HAND_DATASET/VOC_HANDS_DATASET.tar.gz


if __name__ == '__main__':
    data_path = "/home/jade/Data/HANDS_POSE_DATASET/VOC_HANDS_DATASET"
    xml_list = Get_All_Files(os.path.join(data_path,DIRECTORY_ANNOTATIONS),".xml")
    # for xml in xml_list:
    #     imagename,shape, bboxes, labels, labels_text, difficult, truncated =  process_Car(xml)
    #     generateXml(imagename,shape, bboxes, labels, labels_text, difficult, truncated)
    for xml in xml_list:
        imagename, shape, bboxes, labels, labels_text, difficult, truncated = process_hand(xml)
    # plt_bboxes(img, classes, scores, bboxes, figsize=(10, 10), linewidth=1.5):
        img = cv2.imread(os.path.join(os.path.join(data_path,DIRECTORY_IMAGES),imagename.split("/")[-1]))
        shape = img.shape
    # visualizations.plt_bboxes(img,labels,labels,bboxes)
        for i in range(len(bboxes)):
            xmin = int(bboxes[i][0] )
            ymin = int(bboxes[i][1] )
            xmax = int(bboxes[i][2] )
            ymax = int(bboxes[i][3])
            r  = bboxes[i]
            thickness = 1
            img = cv2.rectangle(img, (xmin, ymin ),(xmax, ymax),GetRandomColor())
        cv2.imshow("box", img)
        cv2.waitKey(3000)
