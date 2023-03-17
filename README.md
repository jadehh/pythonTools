# pylkd
Python中使用LDK加密狗

> 一共有4个版本,Linux x86_x64 Linux aarch64 Windows X86 Windows X64
## 使用方法
```python
pyldk = PyLdk(JadeLog=JadeLog)
pyldk.get_ldk()
```
## 制作Wheel
```bash
python setup.py --name manylinux1_x86_64 --path  pyldk/lib/Linux/x86_64 sdist bdist_wheel 

```

{dist}-{version}(-{build})?-{python.version}-{abi}-{platform}.whl