#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : cal_map.py
# @Author   : jade
# @Date     : 2022/9/9 10:08
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
import os
from detector import Detector
from dataset_tools.jade_voc_datasets import GetXmlClassesNames,ProcessXml
from opencv_tools.jade_visualize import *
from opencv_tools.jade_opencv_process import *
from jade import *
import copy
import matplotlib.pyplot as plt
min_overlap = 0.5
font_size = 24
def cal_iou(bbgt,bb,gt_label,pd_label,class_name):
    ovmax = min_overlap
    ov = 0
    wrong_class_status = False
    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
    iw = bi[2] - bi[0] + 1
    ih = bi[3] - bi[1] + 1
    if iw > 0 and ih > 0:
        # compute overlap (IoU) = area of intersection / area of union
        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                          + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
        ov = iw * ih / ua
        if ov > ovmax:
            if pd_label == gt_label == class_name:
                pass
            else:
                ov = 0
                wrong_class_status = True
        else:
            ov = 0

    return ov,wrong_class_status

def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre

def match_boxes(gt_preds,pd_preds,class_name):
    gt_boxes = gt_preds["boxes"]
    gt_labels = gt_preds["labels"]

    pd_boxes = pd_preds["boxes"].tolist()
    pd_labels = pd_preds["labels"]
    ov_iou_list = []
    match_class_status_list = []
    for (i,gt_box) in enumerate(gt_boxes):
        for (j,pd_box) in enumerate(pd_boxes):
            ov_iou,match_class_status = cal_iou(gt_box,pd_box,gt_labels[i],pd_labels[j],class_name)
            if ov_iou > 0:
                ov_iou_list.append(ov_iou)
                match_class_status_list.append(match_class_status)
                gt_boxes.remove(gt_box)
                pd_boxes.remove(pd_box)
    return ov_iou_list,match_class_status_list
from opencv_tools import ReadChinesePath
def cal_map(root_path,detector,is_test=True,show_animation=True):
    sum_AP = 0.0
    n_classes = 0
    label_txt_path = os.path.join(root_path,"label_list.txt")
    if is_test:
        data_txt_path = os.path.join(root_path,"test.txt")
    else:
        data_txt_path = os.path.join(root_path,"train.txt")
    class_name_list = []
    gt_counter_per_class = {}
    no_label_samples_list = []
    with open("output.txt", "w") as output_file:
        with open(label_txt_path, "rb") as f:
            for class_name_byte in f.readlines():
                nd = 0
                n_classes = n_classes + 1
                with open(data_txt_path, "rb") as f2:
                    for data_byte in f2.readlines():
                        class_name = str(class_name_byte, encoding="utf-8").strip()
                        data_list = str(data_byte, encoding="utf-8").strip().split(" ")
                        image_path = os.path.join(root_path, data_list[0])
                        anno_path = os.path.join(root_path, data_list[1])
                        anno_class_names = GetXmlClassesNames(anno_path)
                        if anno_class_names:
                            if class_name in anno_class_names:
                                nd = nd + 1
                gt_counter_per_class[str(class_name_byte, encoding="utf-8").strip()] = nd
                tp = [0] * nd  # creates an array of zeros of size nd
                fp = [0] * nd
                print("正在计算{},检测准确率".format(str(class_name_byte, encoding="utf-8").strip()))
                index = 0
                with open(data_txt_path, "rb") as f2:
                    for data_byte in f2.readlines():
                        class_name = str(class_name_byte, encoding="utf-8").strip()
                        data_list = str(data_byte, encoding="utf-8").strip().split(" ")
                        image_path = os.path.join(root_path, data_list[0])
                        anno_path = os.path.join(root_path, data_list[1])
                        anno_class_names = GetXmlClassesNames(anno_path)
                        img = ReadChinesePath(image_path)
                        gt_preds = {}
                        gt_preds_cp = {}
                        over_lay_list = []
                        if anno_class_names:
                            if class_name in anno_class_names:
                                results = detector.predict(img, class_type=class_name)
                                imagename, shape, bboxes, labels_text, labels, difficult, truncated = ProcessXml(
                                    anno_path, is_rate=False)
                                gt_preds["boxes"] = bboxes
                                gt_preds["labels"] = labels_text
                                gt_preds_cp["boxes"] = copy.copy(bboxes)
                                gt_preds_cp["labels"] = labels_text
                                over_lay_list, match_class_status_list = match_boxes(gt_preds, results, class_name)
                                """
                                Draw image to show animation
                                """
                                if show_animation:
                                    bottom_border = 60
                                    height, widht = img.shape[:2]
                                    # colors (OpenCV works with BGR)
                                    white = (255, 255, 255)
                                    light_blue = (255, 200, 100)
                                    green = (0, 255, 0)
                                    light_red = (30, 30, 255)
                                    # 1st line
                                    margin = 10
                                    v_pos = int(height - margin - (bottom_border / 2.0))
                                    text = "Image: " + GetLastDir(image_path) + " "
                                    img = Add_Chinese_Label(img, text, (margin, v_pos - 48), color=white,
                                                            font_size=font_size)
                                    text = "Class Name : " + class_name + " "
                                    img = Add_Chinese_Label(img, text, (margin, v_pos - 24), light_blue,
                                                            font_size=font_size)
                                    if len(over_lay_list) > 0:
                                        color = light_red
                                        for ovmax in over_lay_list:
                                            width = font_size * len(text)
                                            text = "IoU: {0:.2f}% ".format(ovmax * 100) + "> {0:.2f}% ".format(
                                                min_overlap * 100)
                                            img = Add_Chinese_Label(img, text, (margin + width, v_pos - 24), color,
                                                                    font_size=font_size)
                                    for score in results["scores"]:
                                        text = "confidence: {0:.2f}% ".format(float(score) * 100)
                                        img = Add_Chinese_Label(img, text, (margin, v_pos), white, font_size=font_size)
                                    color = light_red
                                    text = "Result: Match"
                                    if len(over_lay_list) != len(gt_preds_cp["boxes"]):
                                        fp[index] = 1
                                    else:
                                        tp[index] = 1
                                    for match in match_class_status_list:
                                        if match:
                                            text = "Result: Not Match"
                                            tp[index] = 0
                                    img = visualize(img, results)
                                    for (label, box) in zip(gt_preds_cp["labels"], gt_preds_cp["boxes"]):
                                        if label == class_name:
                                            bb = [int(i) for i in box]
                                            img = cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)

                                    line_width = len(text) * font_size
                                    img = Add_Chinese_Label(img, text, (margin + line_width, v_pos), color)
                                    # show image
                                    cv2.namedWindow("Animation", 0)
                                    cv2.imshow("Animation", img)
                                    cv2.waitKey(20)  # show for 20 ms
                                    index = index + 1
                            else:
                                ##不是此类别
                                pass
                                # print("不是此类别")
                        else:
                            if anno_path not in no_label_samples_list:
                                no_label_samples_list.append(anno_path)
                                ##没有目标区域
                                results = detector.predict(img)
                                print("没有目标区域,目标检测结果为:{},标准的结果为:{},anno path = {}".format(results, anno_class_names,
                                                                                          anno_path))
                                img = visualize(img, results)
                                cv2.namedWindow("ERROR Detection", 0)
                                cv2.imshow("ERROR Detection", img)
                                cv2.waitKey(1)
                print(tp, fp)
                # compute precision/recall
                cumsum = 0
                for idx, val in enumerate(fp):
                    fp[idx] += cumsum
                    cumsum += val
                cumsum = 0
                for idx, val in enumerate(tp):
                    tp[idx] += cumsum
                    cumsum += val
                # print(tp)
                rec = tp[:]
                for idx, val in enumerate(tp):
                    rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
                # print(rec)
                prec = tp[:]
                for idx, val in enumerate(tp):
                    prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
                # print(prec)

                ap, mrec, mprec = voc_ap(rec[:], prec[:])

                sum_AP += ap
                text = "{0:.2f}%".format(
                    ap * 100) + " = " + class_name + " AP "  # class_name + " AP = {0:.2f}%".format(ap*100)
                """
                 Write to output.txt
                """
                rounded_prec = ['%.2f' % elem for elem in prec]
                rounded_rec = ['%.2f' % elem for elem in rec]
                output_file.write(
                    text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")
                print(text)

                plt.plot(rec, prec, '-o')
                # add a new penultimate point to the list (mrec[-2], 0.0)
                # since the last line segment (and respective area) do not affect the AP value
                area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
                plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
                # set window title
                fig = plt.gcf()  # gcf - get current figure
                fig.canvas.set_window_title('AP ' + class_name)
                # set plot title
                plt.title('class: ' + text)
                # plt.suptitle('This is a somewhat long figure title', fontsize=16)
                # set axis titles
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                # optional - set axes
                axes = plt.gca()  # gca - get current axes
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
                # Alternative option -> wait for button to be pressed
                # while not plt.waitforbuttonpress(): pass # wait for key display
                # Alternative option -> normal display
                # plt.show()
                # save the plot
                CreateSavePath("classes")
                fig.savefig("classes/" + class_name + ".png")
                plt.cla()  # clear axes for next plot

            output_file.write("\n# mAP of all classes\n")
            mAP = sum_AP / n_classes
            text = "mAP = {0:.2f}%".format(mAP * 100)
            print(text)


if __name__ == '__main__':
    detector = Detector(r"H:\PycharmProjects\Github\ContainerOCR\models\ContaDetModels\2021-12-03")
    cal_map(r"E:\Data\VOC数据集\箱门检测数据集\ContainVOC",detector,is_test=False)