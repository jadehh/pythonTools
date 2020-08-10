"""Setup script for object_detection."""
import os
import shutil
import sys
from jade import get_anaconda_envs_path
def install():
    envs_path_list = get_anaconda_envs_path()
    for envs_path in envs_path_list:
        if os.path.exists(os.path.join(envs_path, "jade/")):
            shutil.rmtree(os.path.join(envs_path, "jade/"))
        shutil.copytree("jade/", os.path.join(envs_path, "jade/"))
        print("Install to " + envs_path)



if __name__ == '__main__':
    install()