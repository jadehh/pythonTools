#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : jade_visualize.py.py
# @Author   : jade
# @Date     : 2021/6/28 13:40
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from PIL import Image,ImageDraw
import numpy as np
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

def draw_box(im, results):
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
            for color in color_list:
                clsid2color[labels_text[i,]] = color
        color = tuple(clsid2color[labels_text[i,]])

        # draw bbox
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
             (xmin, ymin)],
            width=draw_thickness,
            fill=color)
        text = "{} {:.4f}".format(labels_text[i,], scores[i,])
        tw, th = draw.textsize(text)
        draw.rectangle(
            [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
        draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))
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


def visualize_box_mask(im, results, mask_resolution=14):
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
        im = draw_box(im, results)
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

def visualize(image_file,
              results,
              mask_resolution=14):
    # visualize the predict result
    im = visualize_box_mask(
        image_file,
        results,
        mask_resolution=mask_resolution)
    return  np.array(im)