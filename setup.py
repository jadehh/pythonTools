"""Setup script for object_detection."""
import os
import shutil
import sys
def install():
    for sys_path in sys.path:
        if "site-packages" in sys_path and os.path.isdir(sys_path):
            shutil.copytree("jade/", os.path.join(sys_path, "jade/"))
            print ("Install to "+sys_path)
            break



if __name__ == '__main__':
    install()