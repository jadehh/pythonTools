##  OpencvToolsV1.2.4
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
> 需要删除其他包的缓存egg-info

### 更新日志
* 优化VideoCaptureBaseProcess
* 优化Ascend解码速度
* 优化draw ocr 方法
* 输出相机解码失败原因,输出解码失败详细原因
* 自动安装jade最新版本
* 支持华为Ascend解码
* 日志输出Ascend芯片解码
* 优化Ascend芯片解码
* device类型统一使用ascend
* update CVShowBoxes 方法
* update base64转图片和cv2的方法