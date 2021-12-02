#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : createPaddleOCRDatasets.py
# @Author   : jade
# @Date     : 2021/11/25 16:11
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     : Paddle关键点检测数据集转OCR数据集
from jade import *
import json
import math
import re
class ContaNumber(object):
    def __init__(self):
        self._OWNER_CODES = []
        self._TYPE_CODES = []
        self._code_value_map = {'A': 10, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17, 'H': 18,
                                'I': 19, 'J': 20, 'K': 21, 'L': 23, 'M': 24, 'N': 25, 'O': 26, 'P': 27,
                                'Q': 28, 'R': 29, 'S': 30, 'T': 31, 'U': 32, 'V': 34, 'W': 35, 'X': 36,
                                'Y': 37, 'Z': 38}

    def _get_conta_numeric_arr(self, conta_num):
        numeric_conta = []

        if conta_num is not None and len(conta_num):
            for f in conta_num:
                if f >= 'A' and f <= 'Z':
                    numeric_conta.append(self._code_value_map[f])
                else:
                    numeric_conta.append(ord(f) - ord('0'))

        return numeric_conta

    def _get_check_digit(self, numeric_conta_list):
        assert numeric_conta_list is not None and isinstance(numeric_conta_list,
                                                             list), "numeric_conta_list must be a list"
        assert len(numeric_conta_list) == 10 or len(
            numeric_conta_list) == 11, "numeric_conta_list's length must be 10 or 11"

        check_sum = 0
        for i, d in enumerate(numeric_conta_list):
            if i > 9:
                continue
            check_sum = check_sum + d * math.pow(2, i)
        check_sum = check_sum % 11
        if check_sum == 10:
            check_sum = 0

        return chr(int(check_sum + ord('0')))

    def __get_pattern(self, s):
        p = []

        if len(s) == 4:
            return re.sub('\d', '[A-Z]', s)
        else:
            s = re.sub('\d', '', s)
            for i in range(len(s) + 1):
                t = [a + '{1}' for a in s]
                t.insert(i, '[A-Z]')
                p.append(''.join(t))
                if i != 3:
                    p.append('|')

            return ''.join(p)

    def get_check(self, conta):
        if len(conta) > 10:
            conta = conta[:10]

        return self._get_check_digit(self._get_conta_numeric_arr(conta))

    def check_well_conta(self, conta):
        if conta is not None and len(conta) == 11:
            owner_code = conta[:4]
            if not str(owner_code).isalpha():
                return False

            all_alphabet = re.findall('[A-Z]', conta)
            owner_code = ''.join(all_alphabet)
            if len(owner_code) != 4:
                return False

            if not conta.startswith(owner_code):
                return False

            check_code = self.get_check(conta)
            if check_code == conta[-1]:
                ##判断
                return True

        return False

    def check_well_model(self,conta_model):
        self.model_list = ["G", "V", "B", "S", "R", "H", "U", "P", "T", "A",
                           "K"]  ##箱型校验规则,一共四位数,第三位为英文字母满足一定规则,第四位为0-9的数字
        try:
            if conta_model[2] in self.model_list and conta_model[3].isdigit():
                return True
            else:
                return False
        except Exception:
            return False


    def check_conta(self, conta_num):

        if conta_num is None or not isinstance(conta_num, str) or len(conta_num) < 9:
            return conta_num, 0

        if len(conta_num) > 11:
            conta_num = conta_num[:11]

        all_alphabet = re.findall('[A-Z]', conta_num)
        owner_code = ''.join(all_alphabet)
        if not conta_num.startswith(owner_code) or len(owner_code) < 4:
            if len(owner_code) < 2:
                return conta_num, 0

            split_pos = conta_num.rindex(all_alphabet[-1])

            if len(conta_num) - split_pos - 1 != 7:
                return conta_num, 0

            serial_num = conta_num[-7:-1]
            check_digit = conta_num[-1]

            for try_idx in range(2):
                tmp_owner = owner_code
                # judge the left owner code have digit
                if len(owner_code) <= split_pos <= 4 and try_idx == 0:
                    tmp_owner = conta_num[:split_pos + 1]
                    #	tmp_owner = re.sub('0', 'O', tmp_owner)
                    tmp_owner = re.sub('1', 'I', tmp_owner)
                    tmp_owner = re.sub('2', 'Z', tmp_owner)
                    tmp_owner = re.sub('3', 'B', tmp_owner)
                    tmp_owner = re.sub('5', 'S', tmp_owner)
                    #	tmp_owner = re.sub('7', 'Z', tmp_owner)
                    tmp_owner = re.sub('8', 'B', tmp_owner)

                p = self.__get_pattern(tmp_owner)

                for owner in self._OWNER_CODES:
                    # tmp_o = re.findall('[' + owner_code + ']', owner)
                    tmp_o = re.findall(p, owner)

                    if tmp_o is not None and len(tmp_o) > 0:
                        tmp_conta = owner + serial_num
                        tmp_check = self._get_check_digit(self._get_conta_numeric_arr(tmp_conta))
                        if tmp_check == check_digit:
                            return tmp_conta + tmp_check, 1
        else:
            serial_num = conta_num.replace(owner_code, '')
            if 7 < len(serial_num) < 6:
                return conta_num, 0

            serial_num = serial_num[:6]
            tmp_conta = owner_code + serial_num
            tmp_check = self._get_check_digit(self._get_conta_numeric_arr(tmp_conta))
            conta_num = tmp_conta + tmp_check

            return conta_num, 1

        return conta_num, 0

    def get_match_best_text(self,results):
        max_key_val = 0
        match_best_text = None
        for k in results.keys():
            if results[k]["score"] > max_key_val:
                max_key_val = results[k]["score"]
        ##找出得分最大的
        if max_key_val < 1:
            conta_wrong_number = None
            for k in results.keys():
                if results[k]["score"] == max_key_val:
                    conta_wrong_number = k
            if conta_wrong_number:
                if len(conta_wrong_number) < 8:
                    conta_wrong_number = ""
            return "",conta_wrong_number ,max_key_val
        for k in results.keys():
            if results[k]["score"] == max_key_val:
                match_best_text = k
        return match_best_text,"",max_key_val



class CreatePaddleOCRDatasets(object):
    def __init__(self,root_path,save_path,dataset_type=None):
        self.root_path = root_path
        self.save_path = save_path
        self.conta_check_model = ContaNumber()
        self.dataset_type = dataset_type  ## 数据集类型,如车牌数据集,箱号数据集
        label_list = self.get_label_text_path()
        for label_path in label_list:
            self.createOCRDatasets(label_path)



    def verification_rules(self,res_str):
        if self.dataset_type == "箱号数据集":
            if len(res_str) == 11:
                is_check = self.conta_check_model.check_well_conta(res_str)
            else:
                if len(res_str) == 4:
                    if res_str[-1] == "U":
                        is_check = True
                    else:
                        is_check = self.conta_check_model.check_well_model(res_str)
                elif len(res_str) == 7:
                    if res_str.isdigit():
                        is_check = True
                    else:
                        is_check = False
                else:
                    is_check = False
            if is_check:
                return res_str
            else:
                return None
        else:
            return res_str

    def get_rotate_crop_image(self,img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img = img / 255.0
        img_crop_width = int(max(np.linalg.norm(points[0] - points[1]),
                                 np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(max(np.linalg.norm(points[0] - points[3]),
                                  np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0],
                              [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(img, M, (img_crop_width, img_crop_height),
                                      borderValue=.0)

        return dst_img

    def sorted_boxes(self,dt_boxes):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """
        _boxes = []
        for i in range(dt_boxes.shape[0]):
            dt_box = dt_boxes[i, :, :]
            point_list = list(dt_box)
            ##首先确定左上角的点,按照y轴由大到小排列
            dt_box = (sorted(point_list, key=lambda x: x[1]))
            point1 = dt_box[0]
            point2 = dt_box[1]
            point3 = dt_box[2]
            point4 = dt_box[3]
            if point1[0] > point2[0]:
                point_bottom_right = point1
                point_bottom_left = point2
            else:
                point_bottom_right = point2
                point_bottom_left = point1

            if point3[0] > point4[0]:
                point_top_right = point3
                point_top_left = point4
            else:
                point_top_right = point4
                point_top_left = point3
            box = np.array([point_bottom_left, point_bottom_right, point_top_right, point_top_left])
            _boxes.append(box)
        return np.array(_boxes)


    def get_label_text_path(self):
        file_name_list = os.listdir(self.root_path)
        label_path_list = []
        for filename in file_name_list:
            if ".txt" in filename:
                label_path_list.append(os.path.join(self.root_path,filename))
        return label_path_list


    def createOCRDatasets(self,label_txt_path):
        save_h_path = CreateSavePath(os.path.join(self.save_path, "OCRH"))
        save_v_path = CreateSavePath(os.path.join(self.save_path, "OCRV"))
        istrain = False
        if "train" in label_txt_path:
            istrain = True
        privous_dir = GetPreviousDir(label_txt_path)
        with open(label_txt_path, "r") as f:
            content_list = f.read().split("\n")[:-1]
            index = 0
            for content in content_list:
                save_h_detail_path = CreateSavePath(os.path.join(save_h_path, content.split("/")[0]))
                save_v_detail_path = CreateSavePath(os.path.join(save_v_path, content.split("/")[0]))
                save_h_detail_train_path = CreateSavePath(os.path.join(save_h_detail_path, "train"))
                save_h_detail_test_path = CreateSavePath(os.path.join(save_h_detail_path, "test"))

                save_v_detail_train_path = CreateSavePath(os.path.join(save_v_detail_path, "train"))
                save_v_detail_test_path = CreateSavePath(os.path.join(save_v_detail_path, "test"))
                image_path = os.path.join(privous_dir, content.split("\t")[0])
                image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
                label = json.loads(content.split("\t")[1])
                nBox = len(label)
                wordBBs, txts, txt_tags = [], [], []
                for bno in range(0, nBox):
                    wordBB = label[bno]['points']
                    txt = label[bno]['transcription']
                    wordBBs.append(wordBB)
                    txts.append(txt)
                    if txt == '###':
                        txt_tags.append(True)
                    else:
                        txt_tags.append(False)
                wordBBs = self.sorted_boxes(np.array(wordBBs, dtype=np.float32))
                txt_tags = np.array(txt_tags, dtype=np.bool)
                # image = CVShowKeyPoints(image,wordBBs,waiktKey=-1)
                # cv2.imshow("result2",image)
                for i in range(wordBBs.shape[0]):
                    txt_img = self.get_rotate_crop_image(image, wordBBs[i])
                    h, w = txt_img.shape[0], txt_img.shape[1]
                    image_name = GetLastDir(image_path)[:-4] + "_" + str(uuid.uuid1()) + ".jpg"
                    txt_orignal = txts[i]
                    txt = self.verification_rules(txt_orignal)
                    if txt:
                        if istrain is False:
                            if h < w:
                                cv2.imencode('.jpg', txt_img * 255)[1].tofile(
                                    os.path.join(save_h_detail_test_path, image_name))
                                with open(os.path.join(save_h_detail_path, "rec_gt_test.txt"), "a") as f:
                                    content = "test/" + image_name + "\t   " + txt
                                    f.write(content + "\n")
                            else:
                                txt_img = (txt_img * 255).astype("uint8")
                                txt_img = Image_Roate(txt_img, 270)
                                cv2.imencode('.jpg', txt_img)[1].tofile(
                                    os.path.join(save_v_detail_test_path, image_name))
                                with open(os.path.join(save_v_detail_path, "rec_gt_test.txt"), "a") as f:
                                    content = "test/" + image_name + "\t   " + txt
                                    f.write(content + "\n")
                        else:
                            if h < w:
                                cv2.imencode('.jpg', txt_img * 255)[1].tofile(
                                    os.path.join(save_h_detail_train_path, image_name))
                                with open(os.path.join(save_h_detail_path, "rec_gt_train.txt"), "a") as f:
                                    content = "train/" + image_name + "\t   " + txt
                                    f.write(content + "\n")
                            else:
                                txt_img = (txt_img * 255).astype("uint8")
                                txt_img = Image_Roate(txt_img, 270)
                                cv2.imencode('.jpg', txt_img)[1].tofile(
                                    os.path.join(save_v_detail_train_path, image_name))
                                with open(os.path.join(save_v_detail_path, "rec_gt_train.txt"), "a") as f:
                                    content = "train/" + image_name + "\t   " + txt
                                    f.write(content + "\n")
                    else:
                        if txt_orignal != "difficult":
                            print("txt = {}. pass image path = {}".format(txts[i], image_path))

                index = index + 1

    def createDatasets(self,root_path):
        if os.path.exists(os.path.join(root_path, "rec_gt_train.txt")) is True:
            os.remove(os.path.join(root_path, "rec_gt_train.txt"))
        if os.path.exists(os.path.join(root_path, "rec_gt_test.txt")) is True:
            os.remove(os.path.join(root_path, "rec_gt_test.txt"))
        years = os.listdir(root_path)
        with open(os.path.join(root_path, "rec_gt_train.txt"), "w") as f1:
            for year in years:
                if len(year.split("-")) > 1 and os.path.isdir(os.path.join(root_path, year)):
                    with open(os.path.join(root_path, year, "rec_gt_train.txt"), "r") as f:
                        content_list = (f.read().split("\n"))[:-1]
                        processBar = ProgressBar(len(content_list))
                        for content in content_list:
                            new_c = year + "/" + content
                            f1.write(new_c + "\n")
                            processBar.update()

        with open(os.path.join(root_path, "rec_gt_test.txt"), "w") as f1:
            for year in years:
                if len(year.split("-")) > 1 and os.path.isdir(os.path.join(root_path, year)):
                    if os.path.exists(os.path.join(root_path, year, "rec_gt_test.txt")):
                        with open(os.path.join(root_path, year, "rec_gt_test.txt"), "r") as f:
                            content_list = (f.read().split("\n"))[:-1]
                            processBar = ProgressBar(content_list)
                            for content in content_list:
                                new_c = year + "/" + content
                                f1.write(new_c + "\n")
                                processBar.update()



if __name__ == '__main__':
    CreatePaddleOCRDatasets(root_path="E:\Data\字符检测识别数据集\镇江大港厂内车牌关键点检测数据集",save_path="E:\Data\OCR\镇江大港厂内车牌识别数据集")