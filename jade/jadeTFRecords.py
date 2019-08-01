#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/1 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/1  上午10:11 modify by jade

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf
from object_detection.utils import dataset_util
from jade import *


def dict_to_tf_example(img_path,
                       categoty):
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    key = hashlib.sha256(encoded_jpg).hexdigest()
    width = image.width
    height = image.height
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            img_path.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/class/label': dataset_util.int64_feature(categoty),
    }))
    return example


def CreateClassTFRecorder(classify_path, datasetname):
    output_path = CreateSavePath(os.path.join(GetPreviousDir(classify_path), "TFRecords"))
    output_path = output_path + "/" + datasetname + ".tfrecord"
    writer = tf.io.TFRecordWriter(output_path)
    filename = os.listdir(classify_path)
    for i in range(len(filename)):
        imagepaths = GetAllImagesPath(os.path.join(classify_path, filename[i]))
        for imgpath in imagepaths:
            tf_example = dict_to_tf_example(imgpath, i)
            writer.write(tf_example.SerializeToString())


def dict_voc_to_tf_example(data,
                           dataset_directory,
                           label_map_dict,
                           xml_name,
                           year,
                           ignore_difficult_instances=False,
                           image_subdirectory='JPEGImages'
                           ):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
      dataset_directory: Path to root directory holding PASCAL dataset
      label_map_dict: A map from string label names to integers ids.
      ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).
      image_subdirectory: String specifying subdirectory within the
        PASCAL dataset directory holding the actual image data.

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    data['folder'] = dataset_directory + year
    img_path = os.path.join(data['folder'], image_subdirectory, GetLastDir(xml_name)[:-4] + '.jpg')
    full_path = os.path.join(dataset_directory, img_path)
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    if 'object' in data:
        for obj in data['object']:
            # difficult = bool(int(obj['difficult']))
            # if ignore_difficult_instances and difficult:
            # continue

            difficult_obj.append(int(0))

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append("goods".encode('utf8'))
            # classes_text.append(obj['name'].encode('utf8'))
            classes.append(int(1))
            # classes.append(int(label_map_dict[obj['name']]["id"]))
            truncated.append(0)
            obj['pose'] = "Unspecified"
            poses.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example


def main(FLAGS):
    data_dir = FLAGS.data_dir
    CreateSavePath(GetPreviousDir(FLAGS.output_path))
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    label_map_dict, _ = ReadProTxt(FLAGS.label_map_path, id=False)
    for year in FLAGS.years:
        logging.info('Reading from VOC %s dataset.', year)
        examples_list = GetVOCTrainXmlPath(os.path.join(data_dir, year))
        for idx, example in enumerate(examples_list):
            if idx % 100 == 0:
                print('On image %d of %d' % (idx, len(examples_list)))
            with tf.gfile.GFile(example, 'r') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

            tf_example = dict_voc_to_tf_example(data, FLAGS.data_dir, label_map_dict, example, year,
                                                True)
            writer.write(tf_example.SerializeToString())
    writer.close()


def vocTFRecordShow(tfrecord_path):
    # categories, _ = ReadProTxt("/home/jade/label_map/hand.prototxt")
    with tf.Session() as sess:
        example = tf.train.Example()
        # train_records 表示训练的tfrecords文件的路径
        record_iterator = tf.python_io.tf_record_iterator(path=tfrecord_path)
        for record in record_iterator:
            example.ParseFromString(record)
            f = example.features.feature
            # 解析一个example
            # image_name = f['image/filename'].bytes_list.value[0]
            image_encode = f['image/encoded'].bytes_list.value[0]
            image_height = f['image/height'].int64_list.value[0]
            image_width = f['image/width'].int64_list.value[0]
            xmin = f['image/object/bbox/xmin'].float_list.value
            ymin = f['image/object/bbox/ymin'].float_list.value
            xmax = f['image/object/bbox/xmax'].float_list.value
            ymax = f['image/object/bbox/ymax'].float_list.value
            labels = f['image/object/class/label'].int64_list.value
            text = f['image/object/class/text'].bytes_list.value
            image = io.BytesIO(image_encode)
            labels = list(labels)
            xmin = list(xmin)
            ymin = list(ymin)
            xmax = list(xmax)
            ymax = list(ymax)
            image = Image.open(image)

            image = np.asarray(image)
            bboxes = []
            scores = []
            label_texts = []
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for i in range(len(xmin)):
                if labels[i] != 0:
                    print(labels[i])
                    bboxes.append(
                        (xmin[i] * image_width, ymin[i] * image_height, xmax[i] * image_width, ymax[i] * image_height))
                    scores.append(1)
                    # label_texts.append(categories[labels[i]]["display_name"])

                # label_texts.append("good")
            print("**********************")
            CVShowBoxes(image, bboxes, label_texts, labels, scores=scores, waitkey=0)


def parse_exmp(serial_exmp):
    feats = tf.io.parse_single_example(serial_exmp,
                                       features={'image/encoded': tf.io.FixedLenFeature([], tf.string),
                                                 'image/object/class/label': tf.io.FixedLenFeature([], tf.int64)})
    image = tf.io.decode_image(feats['image/encoded'])
    label = feats['image/object/class/label']
    img = tf.cast(image, tf.float32)
    # height = 224
    # width = 224
    # # # Randomly crop a [height, width] section of the image.随机裁剪
    # distorted_image = tf.image.random_crop(img, [height, width, 3])
    # # Randomly flip the image horizontally.随机翻转
    # distorted_image = tf.image.random_flip_left_right \
    #     (distorted_image)
    # # 改变亮度
    # distorted_image = tf.image.random_brightness(distorted_image,
    #                                              max_delta=63)
    # # 改变对比度
    # distorted_image = tf.image.random_contrast(distorted_image,
    #                                            lower=0.2, upper=1.8)
    # img = tf.image.per_image_standardization(distorted_image)
    # img = tf.reshape(img, [224, 224, 3])
    return  img,label





def get_dataset(fname):
    dataset = tf.data.TFRecordDataset(fname)
    return dataset.map(parse_exmp)

def loadClassifyTFRecord(tfrecord_path,batch_size=32,shuffle=True):
    dataset = get_dataset(tfrecord_path)

    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.repeat().batch(batch_size)
    iterator = iter(dataset)
    return iterator


def classifyTFRecordShow(tfrecord_path):
    print(get_dataset(tfrecord_path))


if __name__ == '__main__':
    # CreateClassTFRecorder("/home/jade/Data/sdfgoods10", "sdfgoods10")
    #
    # flags = tf.app.flags
    # flags.DEFINE_string('data_dir', '/home/jade/Data/StaticDeepFreeze/', 'Root directory to raw PASCAL VOC dataset.')
    # flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
    #                                     'merged set.')
    # flags.DEFINE_string('output_path', '/home/jade/Data/StaticDeepFreeze/1/Tfrecords/WildGoods_Train.tfrecord', 'Path to output TFRecord')
    # flags.DEFINE_string('label_map_path', "/home/jade/label_map/wild_goods.prototxt",
    #                     'Path to label map proto')
    # flags.DEFINE_list('years',  ["2019-03-18_14-11-36"],
    #                     'Path to label map proto')
    # FLAGS = flags.FLAGS
    # main(FLAGS)

    loadClassifyTFRecord("/home/jade/Data/TFRecords/sdfgoods10.tfrecord")
