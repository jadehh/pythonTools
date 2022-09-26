#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : jade_visualize.py
# @Author   : opencv_tools
# @Date     : 2021/11/29 19:49
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :

import numpy as np
import cv2
import math
from PIL import Image, ImageFont, ImageDraw
from opencv_tools.jade_opencv_process import GetRandomColor,ReadChinesePath
from jade import getOperationSystem
def get_color_map_list(num_classes):
    """
    Args:
        num_classes (int): number of class
    Returns:
        color_map (list): RGB color list
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map

def draw_box(im, results,show_score=True,font_path=None,font_size=24):
    """
    Args:
        im (PIL.Image.Image): PIL image
        np_boxes (np.ndarray): shape:[N,6], N: number of box,
                               matix element:[class, score, x_min, y_min, x_max, y_max]
        labels (list): labels:['class1', ..., 'classn']
    Returns:
        im (PIL.Image.Image): visualized image
    """
    np_boxes = results['boxes']
    labels_text = results["labels"]
    scores = results["scores"]
    draw_thickness = min(im.size) // 320
    draw = ImageDraw.Draw(im)
    clsid2color = {}
    color_list = get_color_map_list(len(np_boxes))

    for i in range(np_boxes.shape[0]):
        xmin, ymin, xmax, ymax = np_boxes[i,:]
        w = float(xmax) - float(xmin)
        h = float(ymax) - float(ymin)
        if labels_text[i,] not in clsid2color:
                clsid2color[labels_text[i,]] = color_list[i]
        color = tuple(clsid2color[labels_text[i,]])

        # draw bbox
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
             (xmin, ymin)],
            width=draw_thickness,
            fill=color)
        if show_score:
            text = "{} {:.4f}".format(labels_text[i,], scores[i,])
        else:
            text = "{} ".format(labels_text[i,])
        font = ImageFont.truetype(get_font_path(font_path), font_size, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
        tw, th = draw.textsize(text,font)
        draw.rectangle(
            [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
        draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255),font=font)
    return im


def draw_segm(im,
              np_segms,
              np_label,
              np_score,
              labels,
              alpha=0.7):
    """
    Draw segmentation on image
    """
    w_ratio = .4
    color_list = get_color_map_list(len(labels))
    im = np.array(im).astype('float32')
    clsid2color = {}
    np_segms = np_segms.astype(np.uint8)
    for i in range(np_segms.shape[0]):
        mask, score, clsid = np_segms[i], np_score[i], np_label[i] + 1
        if clsid not in clsid2color:
            clsid2color[clsid] = color_list[clsid]
        color_mask = clsid2color[clsid]
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio * 255
        idx = np.nonzero(mask)
        color_mask = np.array(color_mask)
        im[idx[0], idx[1], :] *= 1.0 - alpha
        im[idx[0], idx[1], :] += alpha * color_mask
        sum_x = np.sum(mask, axis=0)
        x = np.where(sum_x > 0.5)[0]
        sum_y = np.sum(mask, axis=1)
        y = np.where(sum_y > 0.5)[0]
        x0, x1, y0, y1 = x[0], x[-1], y[0], y[-1]
        cv2.rectangle(im, (x0, y0), (x1, y1),
                      tuple(color_mask.astype('int32').tolist()), 1)
        bbox_text = '%s %.2f' % (labels[clsid], score)
        t_size = cv2.getTextSize(bbox_text, 0, 0.3, thickness=1)[0]
        cv2.rectangle(im, (x0, y0), (x0 + t_size[0], y0 - t_size[1] - 3),
                      tuple(color_mask.astype('int32').tolist()), -1)
        cv2.putText(
            im,
            bbox_text, (x0, y0 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3, (0, 0, 0),
            1,
            lineType=cv2.LINE_AA)
    return Image.fromarray(im.astype('uint8'))

def draw_lmk(image, lmk_results):
    draw = ImageDraw.Draw(image)
    for lmk_decode in lmk_results:
        for j in range(5):
            x1 = int(round(lmk_decode[2 * j]))
            y1 = int(round(lmk_decode[2 * j + 1]))
            draw.ellipse(
                (x1 - 2, y1 - 2, x1 + 3, y1 + 3), fill='green', outline='green')
    return image
def expand_boxes(boxes, scale=0.0):
    """
    Args:
        boxes (np.ndarray): shape:[N,4], N:number of box,
                            matix element:[x_min, y_min, x_max, y_max]
        scale (float): scale of boxes
    Returns:
        boxes_exp (np.ndarray): expanded boxes
    """
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5
    w_half *= scale
    h_half *= scale
    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp

def draw_mask(im, np_boxes, np_masks, labels, resolution=14, threshold=0.5):
    """
    Args:
        im (PIL.Image.Image): PIL image
        np_boxes (np.ndarray): shape:[N,6], N: number of box,
                               matix element:[class, score, x_min, y_min, x_max, y_max]
        np_masks (np.ndarray): shape:[N, class_num, resolution, resolution]
        labels (list): labels:['class1', ..., 'classn']
        resolution (int): shape of a mask is:[resolution, resolution]
        threshold (float): threshold of mask
    Returns:
        im (PIL.Image.Image): visualized image
    """
    color_list = get_color_map_list(len(labels))
    scale = (resolution + 2.0) / resolution
    im_w, im_h = im.size
    w_ratio = 0.4
    alpha = 0.7
    im = np.array(im).astype('float32')
    rects = np_boxes[:, 2:]
    expand_rects = expand_boxes(rects, scale)
    expand_rects = expand_rects.astype(np.int32)
    clsid_scores = np_boxes[:, 0:2]
    padded_mask = np.zeros((resolution + 2, resolution + 2), dtype=np.float32)
    clsid2color = {}
    for idx in range(len(np_boxes)):
        clsid, score = clsid_scores[idx].tolist()
        clsid = int(clsid)
        xmin, ymin, xmax, ymax = expand_rects[idx].tolist()
        w = xmax - xmin + 1
        h = ymax - ymin + 1
        w = np.maximum(w, 1)
        h = np.maximum(h, 1)
        padded_mask[1:-1, 1:-1] = np_masks[idx, int(clsid), :, :]
        resized_mask = cv2.resize(padded_mask, (w, h))
        resized_mask = np.array(resized_mask > threshold, dtype=np.uint8)
        x0 = min(max(xmin, 0), im_w)
        x1 = min(max(xmax + 1, 0), im_w)
        y0 = min(max(ymin, 0), im_h)
        y1 = min(max(ymax + 1, 0), im_h)
        im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
        im_mask[y0:y1, x0:x1] = resized_mask[(y0 - ymin):(y1 - ymin), (
            x0 - xmin):(x1 - xmin)]
        if clsid not in clsid2color:
            clsid2color[clsid] = color_list[clsid]
        color_mask = clsid2color[clsid]
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio * 255
        idx = np.nonzero(im_mask)
        color_mask = np.array(color_mask)
        im[idx[0], idx[1], :] *= 1.0 - alpha
        im[idx[0], idx[1], :] += alpha * color_mask
    return Image.fromarray(im.astype('uint8'))


def visualize_box_mask(im, results, mask_resolution=14,show_score=True,font_path=None,font_size=12):
    """
    Args:
        im (str/np.ndarray): path of image/np.ndarray read by cv2
        results (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                        matix element:[class, score, x_min, y_min, x_max, y_max]
                        MaskRCNN's results include 'masks': np.ndarray:
                        shape:[N, class_num, mask_resolution, mask_resolution]
        labels (list): labels:['class1', ..., 'classn']
        mask_resolution (int): shape of a mask is:[mask_resolution, mask_resolution]
        threshold (float): Threshold of score.
    Returns:
        im (PIL.Image.Image): visualized image
    """
    if isinstance(im, str):
        im = Image.open(im).convert('RGB')
    else:
        im = Image.fromarray(im)
    if 'masks' in results and 'boxes' in results:
        im = draw_mask(
            im,
            results['boxes'],
            results['masks'],
            results["labels"],
            resolution=mask_resolution)
    if 'boxes' in results:
        im = draw_box(im, results,show_score,font_path,font_size)
    if 'segm' in results:
        im = draw_segm(
            im,
            results['segm'],
            results['label'],
            results['score'],
            results["labels"])
    if 'landmark' in results:
        im = draw_lmk(im, results['landmark'])
    return im

def cv_visualize(image,results,show_score=True,font_path=None,thickness=2,linetype=2,font_size=24):
    if isinstance(image, str):
        image = ReadChinesePath(image)
    boxes = results["boxes"]
    scores = results["scores"]
    labels = results["labels"]
    colors = [GetRandomColor()] * labels.shape[0]
    for (box,score,label,color) in zip(boxes,scores,labels,colors):
        if label != -1:
            cv2.line(image, (int(box[0]), int(box[1])), (int(box[0]), int(box[1] + box[3])), color, thickness,
                     thickness)
            cv2.line(image, (int(box[0]), int(box[1] + box[3])), (int(box[2]), int(box[1] + box[3])), color, thickness,
                     thickness)
            cv2.line(image, (int(box[2]), int(box[1] + box[3])), (int(box[2]), int(box[1])), color, thickness,
                     thickness)
            cv2.line(image, (int(box[2]), int(box[1])), (int(box[0]), int(box[1])), color, thickness, thickness)
            if show_score:
                image = Add_Chinese_Label(image, label+" SCORE:{}".format("%.2f"%score), (int(box[0]), int(box[1])), color, font_size, font_path)
            else:
                image = Add_Chinese_Label(image, label, (int(box[0]), int(box[1])), color, font_size, font_path)

    return image


def visualize(image_file,
              results,
              mask_resolution=14,
              show_score=True,font_path=None,font_size=12):
    # visualize the predict result
    im = visualize_box_mask(
        image_file,
        results,
        mask_resolution=mask_resolution,
        show_score=show_score,
        font_path=font_path,
        font_size=font_size)
    return  np.array(im)


def resize_img(img, input_size=600):
    """
    resize img and limit the longest side of the image to input_size
    """
    img = np.array(img)
    im_shape = img.shape
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(input_size) / float(im_size_max)
    im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
    return im



def str_count(s):
    """
    Count the number of Chinese characters,
    a single English character and a single number
    equal to half the length of Chinese characters.

    args:
        s(string): the input of string
    return(int):
        the number of Chinese characters
    """
    import string
    count_zh = count_pu = 0
    s_len = len(s)
    en_dg_count = 0
    for c in s:
        if c in string.ascii_letters or c.isdigit() or c.isspace():
            en_dg_count += 1
        elif c.isalpha():
            count_zh += 1
        else:
            count_pu += 1
    return s_len - math.ceil(en_dg_count / 2)



# OCR识别结果
def draw_ocr(image, boxes, txts, scores,font_path, draw_txt=True, drop_score=0.5):
    """
    Visualize the results of OCR detection and recognition
    args:
        image(Image|array): RGB image
        boxes(list): boxes with shape(N, 4, 2)
        txts(list): the texts
        scores(list): txxs corresponding scores
        draw_txt(bool): whether draw text or not
        drop_score(float): only scores greater than drop_threshold will be visualized
    return(array):
        the visualized img
    """
    if scores is None:
        scores = [1] * len(boxes)
    for (box, score) in zip(boxes, scores):
        if score < drop_score or math.isnan(score):
            continue
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)

    if draw_txt:
        img = np.array(resize_img(image, input_size=600))
        txt_img = text_visual(
            txts, scores,font_path ,img_h=img.shape[0], img_w=600, threshold=drop_score)
        img = np.concatenate([np.array(img), np.array(txt_img)], axis=1)
        return img
    return image

def text_visual(texts, scores,font_path, img_h=400, img_w=600, threshold=0.):
    """
    create new blank img and draw txt on it
    args:
        texts(list): the text will be draw
        scores(list|None): corresponding score of each txt
        img_h(int): the height of blank img
        img_w(int): the width of blank img
    return(array):

    """
    if scores is not None:
        assert len(texts) == len(
            scores), "The number of txts and corresponding scores must match"

    def create_blank_img():
        blank_img = np.ones(shape=[img_h, img_w], dtype=np.int8) * 255
        blank_img[:, img_w - 1:] = 0
        blank_img = Image.fromarray(blank_img).convert("RGB")
        draw_txt = ImageDraw.Draw(blank_img)
        return blank_img, draw_txt

    blank_img, draw_txt = create_blank_img()

    font_size = 20
    txt_color = (0, 0, 0)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    gap = font_size + 5
    txt_img_list = []
    count, index = 1, 0
    for idx, txt in enumerate(texts):
        index += 1
        if scores[idx] < threshold or math.isnan(scores[idx]):
            index -= 1
            continue
        first_line = True
        while str_count(txt) >= img_w // font_size - 4:
            tmp = txt
            txt = tmp[:img_w // font_size - 4]
            if first_line:
                new_txt = str(index) + ': ' + txt
                first_line = False
            else:
                new_txt = '    ' + txt
            draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
            txt = tmp[img_w // font_size - 4:]
            if count >= img_h // gap - 1:
                txt_img_list.append(np.array(blank_img))
                blank_img, draw_txt = create_blank_img()
                count = 0
            count += 1
        if first_line:
            new_txt = str(index) + ': ' + txt + '   ' + '%.3f' % (scores[idx])
        else:
            new_txt = "  " + txt + "  " + '%.3f' % (scores[idx])
        draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
        # whether add new blank img or not
        if count >= img_h // gap - 1 and idx + 1 < len(texts):
            txt_img_list.append(np.array(blank_img))
            blank_img, draw_txt = create_blank_img()
            count = 0
        count += 1
    txt_img_list.append(np.array(blank_img))
    if len(txt_img_list) == 1:
        blank_img = np.array(txt_img_list[0])
    else:
        blank_img = np.concatenate(txt_img_list, axis=1)
    return np.array(blank_img)




def draw_text_list(img,font_path, label_list, pt_list=[], color_list=[], font_size_list=[]):
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    pilimg = Image.fromarray(cv2img)
    # PIL图片上打印汉字
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    for (label, pt, color, font_size) in zip(label_list, pt_list, color_list, font_size_list):
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
        draw.text(pt, label, color, font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
    cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    return cv2charimg


def get_font_path(font_path):
    if font_path is None:
        if getOperationSystem() == "Windows":
            font_path = r'C:\Windows\Fonts\simhei.ttf'
        else:
            font_path = r'/usr/fonts/simhei.ttf'
        return font_path
    else:
        return font_path

# 添加中文label
def Add_Chinese_Label(img, label, pt1=(0, 0), color=GetRandomColor(), font_size=24,font_path=None):
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    pilimg = Image.fromarray(cv2img)
    # PIL图片上打印汉字
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    font = ImageFont.truetype(get_font_path(font_path), font_size, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
    if type(label) == list:
        for (txt,pt,txt_color) in zip(label,pt1,color):
            draw.text(pt, txt, (int(txt_color[0]), int(txt_color[1]), int(txt_color[2])), font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
    else:
        draw.text(pt1, label, (int(color[0]), int(color[1]), int(color[2])), font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
    cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    return cv2charimg

# 图片加标题，有黑边
def Add_Title_Image(image, title,font_path=None):
    image_shape = image.shape
    image1 = Image.new("RGB", (image_shape[1], image_shape[0]))
    image2 = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image2 = image2.resize((image_shape[1] - 5 * 2, image_shape[0] - 50 * 2), )
    # image.thumbnail(size)
    draw = ImageDraw.Draw(image1)
    # use a truetype font
    font = ImageFont.truetype(get_font_path(font_path), 40)
    draw.text((100, 0), title, font=font)
    bw, bh = image1.size
    lw, lh = image2.size
    image1.paste(image2, (bw - lw, int((bh - lh) / 2)))
    img = cv2.cvtColor(np.asarray(image1), cv2.COLOR_RGB2BGR)
    return img

def draw_text_det_res(img, dt_boxes, txts=None,font_path=None):
    if isinstance(img, str):
        src_im = cv2.imread(img)
    else:
        src_im = img.copy()
    if txts is None:
        txts = [None] * len(dt_boxes)
    for idx, (box, txt) in enumerate(zip(dt_boxes, txts)):
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        if txt:
            src_im = draw_text_list(src_im,[txt[0]], [(box[0, 0], box[3, 1])], [(0, 0, 255)], font_size_list=[54],font_path=font_path)
        cv2.polylines(src_im, [box], True, color=(0, 0, 255), thickness=2)
    return src_im


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

# OCR识别结果
def draw_ocr(image, boxes, txts, scores, draw_txt=True, drop_score=0.5):
    """
    Visualize the results of OCR detection and recognition
    args:
        image(Image|array): RGB image
        boxes(list): boxes with shape(N, 4, 2)
        txts(list): the texts
        scores(list): txxs corresponding scores
        draw_txt(bool): whether draw text or not
        drop_score(float): only scores greater than drop_threshold will be visualized
    return(array):
        the visualized img
    """
    if scores is None:
        scores = [1] * len(boxes)
    for (box, score) in zip(boxes, scores):
        if score < drop_score or math.isnan(score):
            continue
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)

    if draw_txt:
        img = np.array(resize_img(image, input_size=600))
        txt_img = text_visual(
            txts, scores, img_h=img.shape[0], img_w=600, threshold=drop_score)
        img = np.concatenate([np.array(img), np.array(txt_img)], axis=1)
        return img
    return image