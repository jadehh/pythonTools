#coding=utf-8
from jade import *
import cv2
import random
import shutil
color = (255, 255, 255)
drawing = False
size = 5
width = 256


#去除没有标注的文件
def RemoveFilesWithNoLabels(dir,savedir):
    JPEGImages_path = os.path.join(dir,DIRECTORY_IMAGES)
    ANNOTATIONS_path = os.path.join(dir,DIRECTORY_ANNOTATIONS)
    CreateSavePath(os.path.join(savedir,DIRECTORY_IMAGES))
    CreateSavePath(os.path.join(savedir,DIRECTORY_ANNOTATIONS))

    xml_list = GetFilesWithLastNamePath(ANNOTATIONS_path,".xml")
    processbar = ProcessBar()
    processbar.count = len(xml_list)
    for xml in xml_list:
        processbar.start_time = time.time()
        imagename, shape, bboxes, labels, labels_text, difficult, truncated = ProcessXml(xml)
        if len(bboxes) > 0:
            shutil.copy(xml,os.path.join(savedir,DIRECTORY_ANNOTATIONS,GetLastDir(xml)))
            shutil.copy(os.path.join(JPEGImages_path,imagename),os.path.join(savedir,DIRECTORY_IMAGES,imagename))
        NoLinePrint("remove null files ...",processbar)
        
def type_to_chinese(label_types,prototxt_path):
    dicts, class_names = ReadProTxt(prototxt_path)
    labels_text_gt = []
    for label_type in label_types:
        a = dicts[int(class_names.index(label_type))]
        labels_text_gt.append(a["display_name"])
    return labels_text_gt

#通过标注框来裁剪图片
def CutImagesWithBox(VocPath,savedir=None,use_chinese_name=True,prototxt_path="/home/jade/Data/StaticDeepFreeze/ThirtyTypes.prototxt"):

    if savedir is  None:
        savedir = os.path.join(VocPath,"CutImages")
    CreateSavePath(savedir)
    JPEGImages_path = OpsJoin(VocPath,DIRECTORY_IMAGES)
    ANNOTATIONS_path = OpsJoin(VocPath,DIRECTORY_ANNOTATIONS)
    xml_list = GetFilesWithLastNamePath(ANNOTATIONS_path,".xml")
    processbar = ProcessBar()
    processbar.count = len(xml_list)
    for xml in xml_list:
        processbar.start_time = time.time()
        imagename, shape, bboxes, labels, labels_type_text, difficult, truncated = ProcessXml(xml)
        if use_chinese_name:
            labels_text = type_to_chinese(labels,prototxt_path)
        else:
            labels_text = labels
        for i in range(len(bboxes)):
            box = bboxes[i]
            image = cv2.imread(os.path.join(JPEGImages_path,imagename))
            cut_image = image[box[1]:box[3],box[0]:box[2],:]
            cut_image = cv2.resize(cut_image,(width,width))
            name = labels_text[i]
            CreateSavePath(os.path.join(savedir,name))
            cv2.imwrite(os.path.join(savedir,name,imagename[:-4]+"_"+str(i)+".png"),cut_image)
        NoLinePrint("cut image with box ...",processbar)

#恢复裁剪框到VOC
def RestoreCutImageWithVoc(Voc_path,Cut_path):
    image_labels = GetLabelAndImagePath(Cut_path)
    JPEGImages_path = OpsJoin(Voc_path,DIRECTORY_IMAGES)
    ANNOTATIONS_path = OpsJoin(Voc_path,DIRECTORY_ANNOTATIONS)
    xml_list = GetFilesWithLastNamePath(ANNOTATIONS_path,".xml")
    processbar = ProcessBar()

    processbar.count = len(xml_list)
    for xml in xml_list:
        processbar.start_time = time.time()
        imagename, shape, bboxes, labels, labels_type_text, difficult, truncated = ProcessXml(xml)
        image = cv2.imread(os.path.join(JPEGImages_path,imagename))
        new_shape = image.shape
        new_labels = []
        for i in range(len(bboxes)):
            label = image_labels[imagename[:-4]+"_"+str(i)]
            new_labels.append(label)

        GenerateXml(GetLastDir(imagename[:-4]),new_shape,bboxes,new_labels,OpsJoin(Voc_path,"Annotations_new"))
        NoLinePrint("cut image with box ...",processbar)

#保存有Mask的数据
def SaveEraseMask(dir,savedir):
    create_save_path(os.path.join(savedir,"mask"))
    create_save_path(os.path.join(savedir,"image"))
    image_paths = Get_All_Images(dir)
    for image_path in image_paths:
        test_mask = LoadEraseMask(image_path)
        cv2.imwrite(os.path.join(savedir,"mask",GetLastDir(image_path)[:-4]+".png"),test_mask)
        cv2.imwrite(os.path.join(savedir, "image", GetLastDir(image_path)[:-4] + ".png"), cv2.resize(cv2.imread(image_path),(JADE_RESIZE_SIZE,JADE_RESIZE_SIZE)))

#保存随机mask的数据
def SaveRandomMask(dir,savedir):
    create_save_path(os.path.join(savedir,"mask"))
    create_save_path(os.path.join(savedir,"image"))
    image_paths = GetAllImagesPath(dir)
    mask_paths = GetAllImagesPath("/home/jade/Downloads/mask/testing_mask_dataset")
    processbar = ProcessBar()
    processbar.count = len(image_paths) * 50
    for image_path in image_paths:
        random_mask_paths = random.sample(mask_paths, 100)  #从list中随机获取5个元素，作为一个片断返回
        for mask_path in random_mask_paths:
            processbar.start_time = time.time()
            test_mask = cv2.imread(mask_path)
            test_mask = cv2.resize(test_mask,(JADE_RESIZE_SIZE,JADE_RESIZE_SIZE))
            cv2.imwrite(os.path.join(savedir,"mask",GetLastDir(image_path)[:-4]+"_" +GetLastDir(mask_path)[:-4] + ".png"),test_mask)
            cv2.imwrite(os.path.join(savedir, "image", GetLastDir(image_path)[:-4]+"_" +GetLastDir(mask_path)[:-4] + ".png"), cv2.resize(cv2.imread(image_path),(JADE_RESIZE_SIZE,JADE_RESIZE_SIZE)))
            NoLinePrint("Save image ...",processbar)



#将带有mask的图片恢复到原图上
def RestoreRandomMaskWithVOC(dir, savedir):
    voc_path = "/home/jade/Data/Deep_Freeze/jj_VOC"
    voc_jpegimage_path = os.path.join(voc_path,DIRECTORY_IMAGES)
    voc_xml_path = os.path.join(voc_path,DIRECTORY_ANNOTATIONS)
    CreateSavePath(os.path.join(savedir,DIRECTORY_IMAGES))
    CreateSavePath(os.path.join(savedir, DIRECTORY_ANNOTATIONS))
    CreateSavePath(os.path.join(savedir,"image"))
    CreateSavePath(os.path.join(savedir, "mask"))
    image_paths = GetAllImagesPath(dir)
    mask_paths = GetAllImagesPath("/home/jade/Downloads/mask/testing_mask_dataset")
    processbar = ProcessBar()
    cnt = 1
    processbar.count = len(image_paths) * cnt
    for image_path in image_paths:
        processbar.start_time = time.time()
        box_index = int(GetLastDir(image_path).split("_")[1][:-4])
        orignal_image = cv2.imread(os.path.join(voc_jpegimage_path,GetLastDir(image_path).split("_")[0]+".jpg"))
        imagename, shape, bboxes, labels, labels_text, difficult, truncated = ProcessXml(os.path.join(voc_xml_path,GetLastDir(image_path).split("_")[0]+".xml"))
        box = bboxes[box_index]
        random_mask_paths = random.sample(mask_paths, cnt)  # 从list中随机获取5个元素，作为一个片断返回
        for mask_path in random_mask_paths:
            processbar.start_time = time.time()
            test_mask = cv2.imread(mask_path)
            test_mask = cv2.resize(test_mask, (JADE_RESIZE_SIZE, JADE_RESIZE_SIZE))
            image = cv2.resize(cv2.imread(image_path), (JADE_RESIZE_SIZE, JADE_RESIZE_SIZE))
            cv2.imwrite(
                os.path.join(savedir, "image", GetLastDir(image_path)[:-4] + "_" + GetLastDir(mask_path)[:-4] + ".png"),
                image)
            cv2.imwrite(
                os.path.join(savedir, "mask", GetLastDir(image_path)[:-4] + "_" + GetLastDir(mask_path)[:-4] + ".png"),
                test_mask)
            image = mix_image_and_mask(image, test_mask)
            orignal_image[box[1]: box[3], box[0]: box[2],:] = cv2.resize(image,(box[2]-box[0],box[3]-box[1]))
            cv2.imwrite(
                os.path.join(savedir, DIRECTORY_IMAGES, GetLastDir(image_path)[:-4] + "_" + GetLastDir(mask_path)[:-4] + ".png"),
                orignal_image)
            shutil.copy(os.path.join(voc_xml_path,GetLastDir(image_path).split("_")[0]+".xml"),os.path.join(savedir, DIRECTORY_ANNOTATIONS, GetLastDir(image_path)[:-4] + "_" + GetLastDir(mask_path)[:-4] + ".xml"))
            cv2
            NoLinePrint("Save image ...",processbar)



#自定义Mask恢复到原图上
def RestoreEraseMaskWithVOC(dir,savedir):
    CreateSavePath(os.path.join(savedir,DIRECTORY_IMAGES))
    create_save_path(os.path.join(savedir, DIRECTORY_ANNOTATIONS))
    create_save_path(os.path.join(savedir,"image"))
    create_save_path(os.path.join(savedir, "mask"))
    voc_xml_path = os.path.join(dir,DIRECTORY_ANNOTATIONS)
    image_paths = GetAllImagesPath(os.path.join(dir,DIRECTORY_IMAGES))
    mask_paths = GetAllImagesPath("/home/jade/Downloads/mask/testing_mask_dataset")
    mask_paths.sort()
    mask_paths = mask_paths[10000:12000]
    processbar = ProcessBar()
    cnt = 20
    processbar.count = len(image_paths) * cnt
    for image_path in image_paths:
        processbar.start_time = time.time()
        orignal_image = cv2.imread(image_path)
        imagename, shape, bboxes, labels, labels_text, difficult, truncated = ProcessXml(os.path.join(voc_xml_path,GetLastDir(image_path)[:-4]+".xml"))
        for i in range(len(bboxes)):
            box = bboxes[i]
            cut_image = orignal_image[box[1]: box[3], box[0]: box[2], :]
            random_mask_paths = random.sample(mask_paths, cnt)  # 从list中随机获取5个元素，作为一个片断返回
            for mask_path in random_mask_paths:
                orignal_image_copy = orignal_image.copy()
                test_mask = cv2.imread(mask_path)
                test_mask = cv2.resize(test_mask, (JADE_RESIZE_SIZE, JADE_RESIZE_SIZE))
                cut_image_resize = cv2.resize(cut_image, (JADE_RESIZE_SIZE, JADE_RESIZE_SIZE))

                image = mix_image_and_mask(cut_image_resize, test_mask)

                cv2.imwrite(
                    os.path.join(savedir, "image",
                                 GetLastDir(image_path)[:-4] + "_" + str(i) + "_" + GetLastDir(mask_path)[
                                                                                    :-4] + ".png"),
                    image)
                cv2.imwrite(
                    os.path.join(savedir, "mask",
                                 GetLastDir(image_path)[:-4] + "_" + str(i) + "_" + GetLastDir(mask_path)[
                                                                                    :-4] + ".png"),
                    test_mask)
                orignal_image_copy[box[1]: box[3], box[0]: box[2], :] = cv2.resize(image,
                                                                              (box[2] - box[0], box[3] - box[1]))
                cv2.imwrite(
                    os.path.join(savedir, DIRECTORY_IMAGES,
                                 GetLastDir(image_path)[:-4] + "_" + str(i) + "_" + GetLastDir(mask_path)[
                                                                                    :-4] + ".png"),
                    orignal_image_copy)


                shutil.copy(os.path.join(voc_xml_path, GetLastDir(image_path)[:-4] + ".xml"),
                            os.path.join(savedir, DIRECTORY_ANNOTATIONS,
                                         GetLastDir(image_path)[:-4] + "_" + str(i) + "_" + GetLastDir(mask_path)[
                                                                                            :-4] + ".xml"))
                No_Line_Print("Save image ...", processbar)



def SplitDataset(dir,save_dir):
    split_rate = 0.8
    image_file_path = os.path.join(dir,"image")
    mask_file_path = os.path.join(dir,"mask")
    image_list = Get_All_Images(image_file_path)
    num = len(image_list)
    train_list = image_list[0:int(num * split_rate)]
    val_list = [image_path for image_path in image_list if image_path not in train_list]
    create_save_path(os.path.join(save_dir, "image", "training"))
    create_save_path(os.path.join(save_dir, "image", "validation"))
    create_save_path(os.path.join(save_dir, "mask", "training"))
    create_save_path(os.path.join(save_dir, "mask", "validation"))
    for image_path in train_list:

        shutil.copy(image_path,
                    os.path.join(os.path.join(save_dir, "image","training"), GetLastDir(image_path)))
        shutil.copy(os.path.join(mask_file_path,GetLastDir(image_path)),
                    os.path.join(os.path.join(save_dir, "mask", "training"), GetLastDir(image_path)))




    for image_path in val_list:
        shutil.copy(image_path,
                    os.path.join(os.path.join(save_dir, "image" ,"validation"), GetLastDir(image_path)))
        shutil.copy(os.path.join(mask_file_path,GetLastDir(image_path)),
                    os.path.join(os.path.join(save_dir, "mask", "validation"), GetLastDir(image_path)))


    print("Split Done ....")


def CreatFlist(dir,savedir):
    asm_mask_train_name = "asm_masks_train.flist"
    asm_masks_val_name = "asm_masks_val.flist"
    asm_train_name = "asm_train.flist"
    asm_val_name = "asm_val.flist"

    asm_train_path = os.path.join(dir,"image","training")
    train_list = Get_All_Images(asm_train_path)

    asm_val_path = os.path.join(dir,"image","validation")
    val_list = Get_All_Images(asm_val_path)

    asm_mask_train_path = os.path.join(dir,"mask","training")
    train_mask_list = Get_All_Images(asm_mask_train_path)
    mask_val_path = os.path.join(dir,"mask","validation")
    mask_val_list = Get_All_Images(mask_val_path)


    for image_path in train_list:
        with open(os.path.join(savedir,asm_train_name),"a") as f:
            f.write(image_path+"\n")


    for image_path in train_mask_list:
        with open(os.path.join(savedir,asm_mask_train_name),"a") as f:
            f.write(image_path+"\n")

    for image_path in val_list:
        with open(os.path.join(savedir,asm_val_name), "a") as f:
            f.write(image_path + "\n")

    for image_path in mask_val_list:
        with open(os.path.join(savedir,asm_masks_val_name), "a") as f:
            f.write(image_path + "\n")

    print("create flist Done ..")


def ReloadMask(dir):
    mask_train_path = os.path.join(dir,"training")
    mask_val_path = os.path.join(dir,"validation")
    mask_train_list = Get_All_Images(mask_train_path)
    mask_val_list = Get_All_Images(mask_val_path)
    create_save_path(os.path.join(dir,"training2"))
    create_save_path(os.path.join(dir, "validation2"))
    for mask_image_path in mask_train_list:
        mask_image = cv2.imread(mask_image_path)
        cv2.imwrite(os.path.join(dir,"training2",GetLastDir(mask_image_path)[:-4]+".jpg"),np.zeros([100,100]))
        image = cv2.imread(os.path.join(dir,"training2",GetLastDir(mask_image_path[:-4]+".png")))
    for mask_image_path in mask_val_list:
        mask_image = cv2.imread(mask_image_path)
        mask_image = cv2.cvtColor(mask_image,cv2.COLOR_RGB2GRAY)
        cv2.imwrite(os.path.join(dir, "validation2", GetLastDir(mask_image_path)), mask_image)
    print("Done")


def RestoreInpaintingImages(dir,savedir):
    CreateSavePath(os.path.join(savedir,DIRECTORY_IMAGES))
    CreateSavePath(os.path.join(savedir, DIRECTORY_ANNOTATIONS))
    voc_xml_path = os.path.join(dir,DIRECTORY_ANNOTATIONS)
    image_paths = GetAllImagesPath(os.path.join(dir,DIRECTORY_IMAGES))
    processbar = ProcessBar()
    cnt = 1
    processbar.count = len(image_paths) * cnt
    for image_path in image_paths:
        processbar.start_time = time.time()
        orignal_image = cv2.imread(image_path)
        imagename, shape, bboxes, labels, labels_text, difficult, truncated = ProcessXml(os.path.join(voc_xml_path,GetLastDir(image_path)[:-4]+".xml"))
        for i in range(len(bboxes)):
            box = bboxes[i]
            cut_image = cv2.imread(os.path.join(dir,"inpainting",GetLastDir(image_path)))
            orignal_image[box[1]: box[3], box[0]: box[2], :] = cv2.resize(cut_image,
                                                                          (box[2] - box[0], box[3] - box[1]))
            cv2.imwrite(
                os.path.join(savedir, DIRECTORY_IMAGES,
                             GetLastDir(image_path)),
                orignal_image)
            shutil.copy(os.path.join(voc_xml_path, GetLastDir(image_path)[:-4] + ".xml"),
                        os.path.join(savedir, DIRECTORY_ANNOTATIONS,
                                     GetLastDir(image_path)[:-4] + ".xml"))
            NoLinePrint("Save image ...", processbar)


#视频切割图片
def VideoSegmentationImage(video_path,savepath,fps):
    create_save_path(os.path.join(savepath,DIRECTORY_IMAGES))
    create_save_path(os.path.join(savepath, DIRECTORY_ANNOTATIONS))
    processvideo = processVideo(video_path)
    processvideo.Video_To_Images(save_path=os.path.join(savepath,DIRECTORY_IMAGES),cut_fps=5)


def CreateTestTxt(dir):
    with open(os.path.join(dir,"ImageSets","Main","train_var.txt"),'r') as f:
        train = f.read().split("\n")[:-1]

    Allfiles = GetAllImagesPath(OpsJoin(dir,DIRECTORY_IMAGES))
    test = [file for file in Allfiles if file not in train]



    with open(os.path.join(dir,"ImageSets","Main","test_var.txt",'a')) as f:
        train = f.read().split("\n")[:-1]


def CreateVOCDataset(dir,datasetname):
    root_path = dir
    dataset_name = datasetname
    Annotations = DIRECTORY_ANNOTATIONS
    JPEGImages = DIRECTORY_IMAGES

    if os.path.exists(os.path.join(root_path, "ImageSets", "Main")) is not True:
        os.makedirs(os.path.join(root_path, "ImageSets", "Main"))
    else:
        shutil.rmtree(os.path.join(root_path, "ImageSets", "Main"))
        os.makedirs(os.path.join(root_path, "ImageSets", "Main"))
    Main_path = os.path.join(root_path, "ImageSets", "Main")
    image_files = os.listdir(os.path.join(root_path, JPEGImages))

    train_image_files = random.sample(image_files, int(len(image_files) * 0.9))
    test_image_files = [file for file in image_files if file not in train_image_files]

    for train_image_file in train_image_files:
        with open(os.path.join(Main_path, "train_var.txt"), "a") as f:
        #with open(os.path.join(Main_path, "train.txt"), "a") as f:
            image_file = dataset_name + "/" + JPEGImages + "/" + train_image_file
            xml_file = dataset_name + "/" + Annotations + "/" + train_image_file[:-4] + ".xml"
            filename = train_image_file[:-4]
            f.write(filename + "\n")
            #f.write(image_file + " " + xml_file + "\n")

    for test_image_file in test_image_files:
        with open(os.path.join(Main_path, "test_var.txt"), "a") as f:
            #with open(os.path.join(Main_path, "test.txt"), "a") as f:
            image_file = dataset_name + "/" + JPEGImages + "/" + test_image_file
            xml_file = dataset_name + "/" + Annotations + "/" + test_image_file[:-4] + ".xml"
            filename = test_image_file[:-4]
            f.write(filename + "\n")
            #f.write(image_file + " " + xml_file + "\n")
    
    
    
    for train_image_file in train_image_files:
        with open(os.path.join(Main_path, "train.txt"), "a") as f:
        #with open(os.path.join(Main_path, "train.txt"), "a") as f:
            image_file = dataset_name + "/" + JPEGImages + "/" + train_image_file
            xml_file = dataset_name + "/" + Annotations + "/" + train_image_file[:-4] + ".xml"
            filename = train_image_file[:-4]
            #f.write(filename + "\n")
            f.write(image_file + " " + xml_file + "\n")

    for test_image_file in test_image_files:
        with open(os.path.join(Main_path, "test.txt"), "a") as f:
            #with open(os.path.join(Main_path, "test.txt"), "a") as f:
            image_file = dataset_name + "/" + JPEGImages + "/" + test_image_file
            xml_file = dataset_name + "/" + Annotations + "/" + test_image_file[:-4] + ".xml"
            filename = test_image_file[:-4]
            #f.write(filename + "\n")
            f.write(image_file + " " + xml_file + "\n")

def VideoTOImage(video_path,save_path,fps=5):
    PV = processVideo(video_path)
    if type(save_path) == str:
        CreateSavePath(os.path.join(save_path, DIRECTORY_IMAGES))
        CreateSavePath(os.path.join(save_path, DIRECTORY_ANNOTATIONS))
    else:
        CreateSavePath(os.path.join(save_path[0], DIRECTORY_IMAGES))
        CreateSavePath(os.path.join(save_path[0], DIRECTORY_ANNOTATIONS))
        CreateSavePath(os.path.join(save_path[1], DIRECTORY_IMAGES))
        CreateSavePath(os.path.join(save_path[1], DIRECTORY_ANNOTATIONS))
    PV.Video_To_Images(save_path, isthread=False, cut_fps=fps)


def VOCToImages(voc_root_path,save_path):
    create_save_path(os.path.join(save_path,"images"))
    create_save_path(os.path.join(save_path,"masks"))
    image_path = os.path.join(voc_root_path,DIRECTORY_IMAGES)
    xml_path = os.path.join(voc_root_path,DIRECTORY_ANNOTATIONS)
    xmls = Get_All_Files(xml_path,".xml")
    processbar = ProcessBar()
    processbar.count = len(xmls)
    for xml in xmls:
        processbar.start_time = time.time()
        images = cut_images_with_voc(xml)
        for i in range(len(images)):
            masks = load_random_mask()
            for j in range(len(masks)):
                image = mix_image_and_mask(images[i],masks[j])
                cv2.imwrite(
                    os.path.join(save_path, "images", GetLastDir(xml)[:-4] + "_" + str(i) + "_" + str(j) + ".jpg"),
                    image)
                shutil.copy(masks[j], os.path.join(save_path, "masks", GetLastDir(xml)[:-4] + "_" + str(i) + "_" + str(j) + ".jpg"))

        No_Line_Print("saving images ...",processbar)

if __name__ == '__main__':
    #RestoreInpaintingImages("/home/jade/Data/DeepFreeze/VocDataset/zdjj_voc","/home/jade/Data/DeepFreeze/VocDataset/zdjj_voc_voc_inpainting")

    RestoreRandomMaskWithVOC("/home/jade/Data/Deep_Freeze/jj_VAR_VOC",
                                     "/home/jade/Data/Deep_Freeze/JJ_MASK_VOC2")
    #
    # create_voc_dataset("/home/jade/Data/Deep_Freeze/JJ_MASK_VOC2",
    #                     "JJ_MASK_VOC2")
    #
    # restore_inpainting_images("/home/jade/Data/Deep_Freeze/JJ_MASK_VOC2",
    #                           "/home/jade/Data/Deep_Freeze/JJ_Inpainting_VOC2")

    # create_voc_dataset("/home/jade/Data/Deep_Freeze/JJ_Inpainting_VOC2",
    #                    "JJ_Inpainting_VOC2")

    #cut_images_with_box("/home/jade/Data/binggui/ASM_VOC_TEST","/home/jade/Data/binggui/ASM_TEST/image")

    #erase_mask("/home/jade/Data/binggui/ASM_All/image","/home/jade/Data/binggui/ASM_All/mask")
    #split_dataset("/home/jade/Data/binggui/ASM_All","/home/jade/Data/binggui/ASM")
    #creat_flist("/home/jade/Data/binggui/ASM")
    #reload_mask("/home/jade/Data/binggui/ASM/mask")
