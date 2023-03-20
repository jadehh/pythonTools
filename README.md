# pylkd
Python中使用LDK加密狗

> 一共有4个版本,Linux x86_x64 Linux aarch64 Windows X86 Windows X64
## 安装
```bash
pip install pyldk
```
## 使用方法
```python
pyldk = PyLdk(JadeLog=JadeLog)
status,feature_id = pyldk.login()
pyldk.get_ldk(feature_id)
```
