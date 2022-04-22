#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : jade_voc_datasets.py
# @Author   : jade
# @Date     : 2022/3/9 13:33
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from jade import *
from opencv_tools import *
from xml.dom import minidom
from dataset_tools import *
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
def ProcessXml(xml_path,is_rate=True):
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
        if is_rate is False:
            bboxes.append([float(bbox.find('xmin').text) ,float(bbox.find('ymin').text),float(bbox.find('xmax').text),float(bbox.find('ymax').text)])
        else:
            bboxes.append((float(bbox.find('xmin').text) / float(shape[1]),
                           float(bbox.find('ymin').text) / float(shape[0]),
                           float(bbox.find('xmax').text) / float(shape[1]),
                           float(bbox.find('ymax').text) / float(shape[0])))

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
    width = int(shape[1])
    height = int(shape[0])
    depth = int(shape[2])
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
    f = open(save_xml_path, "wb")
    f.write(doc.toprettyxml(indent="  ").encode("utf-8"))
    f.close()

class JadeVOCDatasets(object):
    def __init__(self,root_path):
        self.root_path = root_path
        super(JadeVOCDatasets, self).__init__()

    def remove_no_labels(self):
        file_list = os.listdir(self.root_path)
        processBar  = ProgressBar(len(file_list))
        for file_name in file_list:
            images_path = os.path.join(self.root_path,file_name,DIRECTORY_IMAGES)
            annos_path = os.path.join(self.root_path,file_name,DIRECTORY_ANNOTATIONS)
            image_list = GetAllImagesPath(images_path)
            for image_path in image_list:
                anno_path = os.path.join(annos_path,GetLastDir(image_path)[:-4]+".xml")
                if os.path.exists(anno_path):
                    imagename,shape, bboxes, labels_text,labels, difficult, truncated = ProcessXml(anno_path)
                    if len(labels_text) == 0:
                        os.remove(image_path)
                else:
                    os.remove(image_path)
            processBar.update()

    def change_labels(self,change_labesl,changed_label):
        file_list = os.listdir(self.root_path)
        processBar  = ProgressBar(len(file_list))
        for file_name in file_list:
            if os.path.isdir(os.path.join(self.root_path,file_name)):
                images_path = os.path.join(self.root_path, file_name, DIRECTORY_IMAGES)
                annos_path = os.path.join(self.root_path, file_name, DIRECTORY_ANNOTATIONS)
                image_list = GetAllImagesPath(images_path)
                for image_path in image_list:
                    anno_path = os.path.join(annos_path, GetLastDir(image_path)[:-4] + ".xml")
                    if os.path.exists(anno_path):
                        imagename, shape, bboxes, labels_text, labels, difficult, truncated = ProcessXml(anno_path,is_rate=False)
                        new_labels_text = []
                        for label_text in labels_text:
                            if label_text in change_labesl:
                                new_labels_text.append(changed_label)
                        GenerateXml(GetLastDir(image_path)[:-4],shape,bboxes,new_labels_text,os.path.join(self.root_path,file_name,DIRECTORY_ANNOTATIONS))
                processBar.update()

    def video_to_voc(self,save_path,detector=None,fps=5):
        video_list = GetFilesWithLastNamePath(self.root_path, ".avi")
        processBar = ProgressBar(len(video_list))
        for video_path in video_list:
            capture = cv2.VideoCapture(video_path)
            index = 0
            while capture.isOpened():
                ret, frame = capture.read()
                if ret is False:
                    break
                if detector is None:
                    if index % fps == 0:
                        WriteChienePath(os.path.join(save_path, GetSeqNumber() + ".jpg"), frame)
                    index = index + 1
            processBar.update()


if __name__ == '__main__':
    jadeVOCDatasets = JadeVOCDatasets(r'F:\数据集\VOC数据集\定制版顶相机箱号检测数据集')
    # jadeVOCDatasets.remove_no_labels()
    jadeVOCDatasets.change_labels(["PCTNNO","NCTNNO"],"CTNNO")