### 更新日志


#### JadeV2.1.2 - 2023-11-20
* zip_package Linux生成文件名修改
---

<details onclose>
<summary>查看更多更新日志</summary>

#### JadeV2.1.1 - 2023-11-20
* zip_package 压缩成压缩包文件名不能为中文,
---

#### JadeV2.1.0 - 2023-11-20
* 拷贝config文件和压缩文件夹,区分Linux和Windows
---

#### JadeV2.0.9 - 2023-11-20
* 拷贝config文件和压缩文件夹
---

#### JadeV2.0.8 - 2023-09-21
* 解决加密狗重复登录的bug
---

#### JadeV2.0.7 - 2023-07-19
* 解决打包一个完成的包,exclude_files不生效的bug
---

#### JadeV2.0.6 - 2023-07-19
* 新增exclude_files参数,去除打包时不必要的动态库
* 解决如果为空,打包失败的bug
---


#### JadeV2.0.5 - 2023-06-14
* 兼容加密狗feature id list为None的情况
---

#### JadeV2.0.4 - 2023-05-22
* 加密狗监测模块支持多个加密狗的id
---


#### JadeV2.0.3 - 2023-05-11
* 打包的时候新增是否压缩lib包参数
---

#### JadeV2.0.2 - 2023-05-08
* 初始化的时候就需要校验feature id 是否存在
---

#### JadeV2.0.1 - 2023-05-08
* 加密狗如果没有feature id或者获取feature id失败时,需要退出会话
---

#### JadeV1.9.9 - 2023-05-08
* 加密狗的监测支持自定义feature id
---

#### JadeV1.9.8 - 2023-04-21
* 加密狗监测线程支持多个加密狗
* 如果当前登录的加密狗超过最大限制数量,在查找是否还有新的加密狗
---

#### JadeV1.9.7 - 2023-04-03
* 如果可执行文件存在的化,则拷贝Output文件夹,如果不存在则不拷贝
---

#### JadeV1.9.6 - 2023-04-03
* 编译的时候也需要将lib包拷贝到release文件夹下
---


#### JadeV1.9.5 - 2023-04-03
* 压缩lib包时,需要判断是否有可执行文件
---


#### JadeV1.9.4 - 2023-04-03
* 不在固定pyinstaller版本为4.6
---


#### JadeV1.9.3 - 2023-03-24
* 固定pyinstaller版本为4.6
---


#### JadeV1.9.2 - 2023-03-24
* 优化加密狗判断流程,刷新的时候一直占用一个Session
---

#### JadeV1.9.1 - 2023-03-24
* pyinstaller版本不固定
---

#### JadeV1.9.0 - 2023-03-24
* 监听加密狗驱动是否在线时间可配置
---


#### JadeV1.8.9 - 2023-03-24
* 新增LDK资源释放函数
---

#### JadeV1.8.8 - 2023-03-24
* 新增监听加密狗是否在线线程,并且线程初始化的时候就启动
---

#### JadeV1.8.7 - 2023-03-24
* 程序被kill获取退出状态,可以做释放资源
---


#### JadeV1.8.6 - 2023-03-22
* 打包模块修复压缩函数找不到的bug
---

#### JadeV1.8.5 - 2023-03-20
* 打包模块支持打包并压缩lib包到Output文件夹中
---

#### JadeV1.8.4 - 2023-03-20
* 新增读取README.md内容方法
---

#### JadeV1.8.3 - 2023-03-17
* 新增获取app version的方法
* 自动写入version文件
* 解决打包wheel的报错
---

#### JadeV1.8.2 - 2023-03-14
* 新增压缩文件夹到指定文件夹下方法
---

#### JadeV1.8.1 - 2023-03-13
* 更新获取版本号方法
---

#### JadeV1.8.0 - 2023-03-10
* Inno Setup 不输出信息
* Xcopy不输出信息
---
#### JadeV1.7.9 - 2023-03-10
* ProgressBar Windows下还是继续使用中文
---
#### JadeV1.7.8 - 2023-03-09
* 修改AppRun文件,解决在github action 自动打包execv error: Permission denied 的问题
---

#### JadeV1.7.7 - 2023-03-08
* 制作AppImage先给可执行文件赋予权限
---
#### JadeV1.7.6 - 2023-03-08
* 解决log level的bug
---
#### JadeV1.7.5 - 2023-03-08
* Release文件使用CONTRIBUTING.md
---

#### JadeV1.7.4 - 2023-03-08
* 先创建Release,在上传文件到Release
---
#### JadeV1.7.3 - 2023-03-08
* 无需上传action附件,直接上传至release
---

#### JadeV1.7.2 - 2023-03-08
* 测试发布使用模糊路径
---

#### JadeV1.7.1 - 2023-03-08
* 打包模块新增log_level参数
* 支持使用Inno Setup打包Windows安装包
---

#### JadeV1.7.0 - 2023-03-08
* 修改Windows下ISS文件生成的路径为当前目录
---

#### JadeV1.6.9 - 2023-03-07
* 解决编译失败,windows下输出的bug
---
#### JadeV1.6.8 - 2023-03-07
* wheel包需要配置version模块
* 打包版本由更新日志统一管理,其他地方无需在配版本号
---

#### JadeV1.6.7 - 2023-03-07
* 修改ChangeLog文件的名称改为CONTRIBUTING
---
#### JadeV1.6.6 - 2023-03-07
* 固定ChangeLog的格式,使Release发布界面美观
---
#### JadeV1.6.5 - 2023-03-07
* 创建Release的时候使用tag名称作为当前Release的名称
---
#### JadeV1.6.4 - 2023-03-07
* 打包的时候自动判断当前版本
* 支持自动打包脚本,解决Windows下Progress的bug
---
#### JadeV1.6.3 - 2023-03-06
* 支持遍历文件夹下所有文件
* 加密文件支持传入输出文件夹名称参数
* 打包时候支持自定义输出名
* 更新模型加密加密方法,支持没有后缀名称的模型加解密
* 固定cryptography版本
* 重新优化packing模块
* ui文件转py文件时，新增是否翻译功能
* 取消main文件没有的输出
* 数据库支持清空操作
* 支持模型解密自定义保存路径
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
* update Linux打包使用原始的版本
---
</details>