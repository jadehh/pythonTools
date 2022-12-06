##  OpencvToolsV1.1.4
Opencv相关操作

### 打包为wheel文件

安装wheel
```bash
pip install wheel
```
打包wheel
```bash
pip wheel --wheel-dir=./wheel_dir ./
```
> wheel-dir 为wheel 输出文件夹，后面接项目文件夹（即包含setup.py的文件夹）

### 更新日志
* 支持华为Ascend解码
* 日志输出Ascend芯片解码
* 优化Ascend芯片解码
