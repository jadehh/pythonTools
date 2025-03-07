# 项目名称

## 简介
该项目包含一个用于与 NVR 设备交互的 Python 客户端。主要功能包括从 NVR 下载录像文件。

## 文件说明

### `main.py`
该文件包含主程序入口，主要功能是初始化 NVR 客户端并下载指定时间段的录像。

### `test.py`
该文件包含测试代码，主要功能是从文件中读取数据并下载对应时间段的录像。

### `src/nvr.py`
该文件包含 NVR 客户端的实现，定义了与 NVR 设备交互的各种方法和结构体。

## 使用方法

### 运行 `main.py`
1. 修改 `main.py` 中的 NVR IP 地址、用户名和密码。
2. 运行 `main.py`：
    ```bash
    python main.py
    ```

### 运行 `test.py`
1. 修改 `test.py` 中的 NVR IP 地址、用户名和密码。
2. 确保 `过卡记录.txt` 文件存在并包含正确的数据格式。
3. 运行 `test.py`：
    ```bash
    python test.py
    ```

## 主要类和方法

### `NVRClient` 类
该类用于与 NVR 设备交互，主要方法包括：

- `connect()`: 初始化并连接到 NVR 设备。
- `download_recording(save_path, start_time_str, end_time_str)`: 下载指定时间段的录像文件。
- `get_channel_numbers()`: 获取通道号。
- `disconnect()`: 断开连接并清理资源。

### 结构体
- `NET_DVR_DEVICEINFO_V30`: 设备信息结构体。
- `NET_DVR_TIME`: 时间结构体。
- `NET_DVR_PLAYCOND`: 播放条件结构体。

## 注意事项
- 确保 HCNetSDK 库文件存在于指定路径。
- 修改代码中的 IP 地址、用户名和密码为实际 NVR 设备的参数。
- 确保保存路径存在或代码中会自动创建。

## 联系方式
如有任何问题，请联系作者：jadehh@1ive.com