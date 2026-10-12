# 错误模式速查

## 环境错误

| 报错信息 | 原因 | 修复 |
|---------|------|------|
| `ASCEND_HOME_PATH not found` | 未 source CANN 环境 | root: `source /usr/local/Ascend/cann/bin/setenv.bash`；普通用户: `source ~/Ascend/cann/bin/setenv.bash` |
| `ImportError: libhccl.so` | CANN 库路径缺失 | source setenv.bash，检查 LD_LIBRARY_PATH |
| `AttributeError: 'NoneType' object has no attribute 'acl_init'` | CANN 环境未初始化 | source setenv.bash + 设置 ASCEND_HOME_PATH |
| Python 版本过低 | 需要 3.8+ | `python3 --version` 确认 |

### CANN 分包环境变量

```shell
# root 用户
source /usr/local/Ascend/cann/bin/setenv.bash

# 普通用户
source ~/Ascend/cann/bin/setenv.bash

# 非标安装路径
export ASCEND_CUSTOM_PATH=/your/custom/ascend/path
source $ASCEND_CUSTOM_PATH/bin/setenv.bash
```

## CSV 解析错误

| 报错信息 | 原因 | 修复 |
|---------|------|------|
| 字段解析失败 | 含特殊字符未加双引号 | `"((2,3),(2,3))"` 而非 `((2,3),(2,3))` |
| dtype 不识别 | 使用了非标准字符串 | 使用 `float32`/`fp32` 等标准名称 |
| shape 元素数量不匹配 | input_shapes 与 input_dtypes 长度不一致 | 检查元组嵌套层级 |

## 编译错误

| 报错信息 | 原因 | 修复 |
|---------|------|------|
| `op not found` | op_name 拼写错误 | 检查算子名称是否与 CANN 注册名一致 |
| tiling 失败 | shape 不支持或属性缺失 | 检查 attributes 字段，确认 shape 合法性 |
| 编译超时 | 算子过大或 `--proc-timeout` 太小 | 减小 shape；或加大 `--proc-timeout` 值（默认 3600 秒） |

## 执行错误

| 报错信息 | 原因 | 修复 |
|---------|------|------|
| 执行超时 | 算子计算量过大 | 加大 `--proc-timeout` 值（默认 3600 秒，未显式设置也可能超时） |
| `exit -9` / `exit -11`（批跑） | 进程复用时内存累积或信号干扰 | 单独执行该用例验证，或加 `--proc-no-reuse` 禁止进程复用 |
| 多卡部分失败 | 设备异常或资源竞争 | `--devices=^2,^3` 排除问题设备，`--pc=1` 减少进程 |

## 插件错误

| 报错信息 | 原因 | 修复 |
|---------|------|------|
| Golden 函数不生效 | 未传 `--plugin` 参数 | 命令行加 `--plugin my_plugin.py` |
| Spec 未匹配 | TestSpec 类名/`__spec__` 注册名与 CSV 不一致 | 确保类名遵循 `PascalCase+TestSpec`（如 `AbsTestSpec`）或 `__spec__` dict 的 key 与 CSV `op_name`/`api_name` 一致 |
| 插件加载报错 | 语法或运行时错误 | 用 `python3 -c "import ast; ast.parse(open('my_plugin.py').read())"` 检查语法 |
| 同名注册 | 多个插件注册相同算子名 | 按搜索顺序先命中者生效（不覆盖、不打印警告，同文件内后者覆盖）；确保注册名称唯一 |
| TestSpec 加载失败 | `__spec__` 文件语法错或 Spec 类定义有误 | 检查 `__spec__` 标记与 Spec 类；语法错会记 error 日志（仅 `__spec__` 文件） |
