#coding=utf-8
import os
file_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
font_path = "{}/jade_simhei.ttf".format(file_path)
DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'JPEGImages/'
DIRECTORY_PREANNOTATIONS = "PredictAnnotations/"
from opencv_tools.jade_visualize import *


