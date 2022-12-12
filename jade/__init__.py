#coding=utf-8
import os
file_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
AppRunPath = "{}/AppRun".format(file_path)
LogoPath = "{}/app_logo.png".format(file_path)
from jade.version import full_version as __version__
from jade.jade_logging import *
from jade.jade_packing import *
from jade.jade_tools import *
from jade.jade_sqlite_data_base import *
from jade.jade_progress_bar import *