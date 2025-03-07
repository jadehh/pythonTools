# 更新说明

## 更新内容
1. 添加了 `NET_DVR_DEVICEINFO_V30` 结构体的 Python 实现。
2. 添加了 `NET_DVR_TIME` 结构体的 Python 实现。
3. 添加了 `NET_DVR_PLAYCOND` 结构体的 Python 实现。
4. 更新了 `README.md` 文件，增加了项目简介、文件说明、使用方法、主要类和方法、注意事项等内容。

## 详细变更
- 在 `src/nvr.py` 文件中，定义了 `NET_DVR_DEVICEINFO_V30`、`NET_DVR_TIME` 和 `NET_DVR_PLAYCOND` 结构体。
- 在 `main.py` 和 `test.py` 文件中，添加了对新结构体的使用示例。
- `README.md` 文件中，详细描述了项目的使用方法和注意事项。

## 注意事项
- 确保 HCNetSDK 库文件存在于指定路径。
- 修改代码中的 IP 地址、用户名和密码为实际 NVR 设备的参数。
- 确保保存路径存在或代码中会自动创建。

## 联系方式
如有任何问题，请联系作者：jadehh@1ive.com