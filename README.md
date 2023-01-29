##  python tools V1.0.6
* 不使用第三方wheel包


## 1.编写python文件

## 2. 编写setup.py文件
```Python
from setuptools import setup, find_packages

setup(
    name="jade_tools",
    version="0.1",
    keywords=("pip", "jade_tools", ""),
    description="jade_tools",
    long_description="xxx",
    license="MIT Licence",

    url="https://jadehh@live.com",
    author="dataset_tools",
    author_email="jadehh@live.com",

    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=["numpy","pillow","imageio"]  # 这个项目需要的第三方库
)
```
## ３．打包为wheel文件

安装wheel
```bash
pip install wheel
```
打包wheel
```bash
python setup.py sdist bdist_wheel

```
> wheel-dir 为wheel 输出文件夹，后面接项目文件夹（即包含setup.py的文件夹）
