"""Setup script for object_detection."""
import os
import shutil
import sys
def install():
    for sys_path in sys.path:
        if "site-packages" in sys_path or 'dist-packages' in sys_path and os.path.isdir(sys_path):
            if os.path.exists(os.path.join(sys_path, "jade/")):
                shutil.rmtree(os.path.join(sys_path, "jade/"))
            shutil.copytree("jade/", os.path.join(sys_path, "jade/"))
            print ("Install to "+sys_path)
            break



if __name__ == '__main__':
    install()