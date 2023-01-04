##  OpencvToolsV1.1.8
Opencv相关操作

### 打包为wheel文件

安装wheel
```bash
pip install wheel
```
打包wheel
```bash
python setup.py sdist bdist_wheel

```
> wheel-dir 为wheel 输出文件夹，后面接项目文件夹（即包含setup.py的文件夹）

### 更新日志
* 自动安装jade最新版本
* 支持华为Ascend解码
* 日志输出Ascend芯片解码
* 优化Ascend芯片解码
* device类型统一使用ascend
* update CVShowBoxes 方法
* update base64转图片和cv2的方法