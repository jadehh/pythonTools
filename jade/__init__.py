#coding=utf-8
COLOR_CLASSES = [[0,0,0],[255,255,0],[176,224,230],[227,207,80],[65,105,225]]*10
JADE_COLOR = (255, 255, 255)
JADE_DRAWING = False
JADE_SIZE = 8
JADE_RESIZE_SIZE = 256

DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'JPEGImages/'
DIRECTORY_PREANNOTATIONS = "PredictAnnotations/"
# COLORS = [(183, 68, 69), (86, 1, 17), (179, 240, 121),
#           (97, 134, 238), (145, 152, 245),
#           (170, 153, 97), (124, 250, 3), (100, 151, 78),
#           (177, 117, 215), (183, 70, 5), (56, 165, 105),
#           (197, 92, 108), (251, 79, 27), (205, 220, 142),
#           (46, 76, 247), (172, 178, 163), (119, 163, 227),
#           (220, 83, 194), (178, 152, 147), (56, 91, 90), (136, 4, 5)]


VOC_LABELS = ('__background__',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')
# VOC_CLASSES = ( '__background__', # always index 0
#     'person')


# VOC_LABELS = {
#     'none': (0, 'Background'),
#     'thumb_up': (1, 'thumb_up'),
#     'others':(2,'others')
#
# }

# VOC_LABELS = {
#     'none': (0, 'Background'),
#     'asm-asmnc-pz-yw-500ml':(1,'asm-asmnc-pz-yw-500ml'),
#     'ty-hzy-pz-gw-500ml':(2,'ty-hzy-pz-gw-500ml'),
#     ('yb-ybcjs-pz-yw-555ml'):(3,'yb-ybcjs-pz-yw-555ml'),
#     ('wlj-wljlc-hz-yw-250ml'):(4,'wlj-wljlc-hz-yw-250ml'),
#     ('wlj-wljlc-gz-yw-310ml'): (21, 'wlj-wljlc-gz-yw-310ml'),
#     ('nfsq-nfsqjjydyl-pz-nmw-550ml'):(5,'nfsq-nfsqjjydyl-pz-nmw-550ml'),
#     ('yy-yylght-gz-ht-240ml'):(6,'yy-yylght-gz-ht-240ml'),
#     ('nfsq-nfsqyytrs-pz-yw-550ml'):(7,'nfsq-nfsqyytrs-pz-yw-550ml'),
#     ('mzy-mzyglny-pz-blw-450ml'):(8,'mzy-mzyglny-pz-blw-450ml'),
#     ('ksf-ksfmlqc-pz-yw-500ml'):(9,'ksf-ksfmlqc-pz-yw-500ml'),
#     ('ksf-ksfmlmc-pz-yw-500ml'):(10,('ksf-ksfmlmc-pz-yw-500ml')),
#     'hand':(20,'hand'),
#     'mask':(11,'mask')
# }




#coding=utf-8
from jade.jade_tools import *
from jade.jade_processfile import *
from jade.ReadVocData import *
from jade.jade_image_processing import *
from jade.clean_dataset import *

