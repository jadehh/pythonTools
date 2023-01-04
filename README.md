##  python tools V1.5.4
* 不使用第三方wheel包

## 更新日志
* ui文件转py文件时，新增是否翻译功能
* 取消main文件没有的输出
* 数据库支持清空操作
* 支持模型解密自定义保存路径

<details onclose>
<summary>查看更多更新日志</summary>

* packing模块支持导入其他库传参
* 优化jade packing 模块
* 新增version文件
* update jade packing 支持自定义main函数文件
* update getSectionConfigs 方法
* 版本号支持4位版本号和3位版本号
* update 支持模型加密解密操作,支持解密直接返回字节流
* 引入新的cffi包
* update 支持不使用图片进行打包
* update 支持python3.7进行打包
* update 路径转换无需判断路径是否真实存在
* update 打包的时候支持文件夹导入
* update 打包成AppImage时候无需icon图标
* update 加入AppImage图标为默认图标
* update Linux打包使用原始的版本号
</details>


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
    author="jade",
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