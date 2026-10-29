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
| GEIR `EZ1001` | `concat_d`（DYNAMIC_INPUT 聚合）在 `-d` 动态编译下 GE 侧 shape 校验失败（已知限制） | 该用例改用 `-c` 静态编译；其余 TensorList 算子全模式可用 |
| GEIR 编译 C++ 源码失败 / 无 `g++` | 系统缺工具链（GEIR 生成 C++ 源码后需 `g++` 编译） | `g++ --version` 确认；缺失时 `yum install gcc-c++` 或 `apt install g++` |
| GEIR 动态 shape 未生效 | GEIR 默认开启静态编译 `-c`，与 Kernel 默认 `-d` 相反；`dyn_input_shapes` 仅在 `-d` 下生效 | 显式传 `-d`（可 `-c=false -d` 仅动态） |

## 执行错误

| 报错信息 | 原因 | 修复 |
|---------|------|------|
| 执行超时 | 算子计算量过大 | 加大 `--proc-timeout` 值（默认 3600 秒，未显式设置也可能超时） |
| `exit -9` / `exit -11`（批跑） | 进程复用时内存累积或信号干扰 | 单独执行该用例验证，或加 `--proc-no-reuse` 禁止进程复用 |
| 多卡部分失败 | 设备异常或资源竞争 | `--devices=^2,^3` 排除问题设备，或 `--pc=1` 减少进程 |

## E2E ONNX 通路错误

onnx（`onnx.<OpType>`）链路：TTK 组模 → atc 编译 .om → aclmdl 执行。专属失败码：

| 报错码 | 原因 | 修复 |
|--------|------|------|
| `ONNX_NPUSIM_UNSUPPORTED` | `--backend npusim` + onnx 用例（启动期整批拒绝） | 去掉 npusim 用 `--backend npu`（om 执行需真机：atc + aclmdl），或该算子改用 kernel/aclnn 模式 |
| `ONNX_XPU_UNSUPPORTED` | `--golden-mode XPU` / `--xpu-perf` / `--compare cross_check` + onnx 用例（启动期整批拒绝） | 去掉这些开关（三方远端按框架 dotted path 分发，无 onnx 通道）；`--dump xpu` 自动告警跳过 |
| `ATC_NOT_FOUND` | 未找到 atc | 安装 CANN 工具链并 source set_env.sh（或经 `ASCEND_TOOLKIT_HOME` 让 TTK 自动探测） |
| `ATC_COMPILE_FAILED` | atc 编译失败（stdout+stderr 尾部原文透传） | 看透传的 atc 日志定位：常见为算子/shape/属性/dtype 不支持或 opset 超限 |
| `OM_LOAD_FAILED` / `OM_EXECUTE_FAILED` | om 加载、desc 查询或 aclmdl 执行失败 | 查日志中透传的 ACL 错误信息；确认环境 libascendcl 正常 |
| `ONNX_DYNAMIC_OUTPUT_UNSUPPORTED` | 数据依赖 shape 的算子（NonZero/TopK 等）产出动态 dim om | 通路边界，此类算子改用 kernel/aclnn 模式覆盖 |
| `ONNX_DYNAMIC_GRAPH_UNSUPPORTED` | onnx不支持 `-d` 动态图 | 去掉 `-d`（onnx 的 eager ≡ 静态图，om 恒为静态图） |
| `ONNX_DYNAMIC_INPUT_UNSUPPORTED` | `tensor_view_shapes` 输入槽含 dim<0（-1 动态标记） | onnx仅支持静态 shape（dim ≥ 0）；去掉 -1 槽，动态 shape 覆盖改用 kernel/aclnn 模式 |
| `ONNX_ACLGRAPH_UNSUPPORTED` | onnx不支持 aclgraph 模式 | 去掉 `--aclgraph` |
| `ONNX_LAYOUT_UNSUPPORTED` | onnx 用例填写了布局列（formats/storage_shapes/strides/offsets） | 清空这些列（onnx仅 ND 连续语义） |
| `ONNX_OPSET_VERSION_UNSUPPORTED` | 环境 onnx 包 opset 版本 > atc 插件注册上限（18） | 安装满足 `onnx.defs.onnx_opset_version() ≤ 18` 的 onnx 包（如 onnx 1.13.x 配 CANN 9.2.x） |
| `ONNX_DOMAIN_UNSUPPORTED` | api_name 带 domain 限定（非纯两段） | 改写 `onnx.<OpType>`（不带 domain/opset 限定） |
| `ONNX_ATTR_TYPE_UNSUPPORTED` | 算子 schema 声明了 TENSOR/GRAPH 载荷属性，无法从 CSV 标量构造 | 该类算子不支持，换其他模式覆盖 |
| `ONNX_OPTIONAL_INPUT_UNSUPPORTED` | Optional 输入非尾部省略 | 调整 `tensor_view_shapes` 使省略的 None 槽只在尾部 |
| `ONNX_CPU_EXECUTION_UNSUPPORTED` | `--backend cpu` 下执行 onnx 用例（om 执行需 NPU） | 去掉 `--backend cpu`；CPU 侧仅 golden（ReferenceEvaluator）与输入生成 |
| `UNKNOWN_ATTRIBUTE`（结果列 fail_reason） | attributes 列出现 schema 未声明的属性名 | 按环境 onnx schema 核对拼写（onnx不静默忽略未知属性） |

> atc 编译缓存位于 `<root>/onnx_cache/<op_type>/`（缓存键含 atc ONNX 插件 .so 指纹，插件热替换自动失效）；怀疑缓存脏时可删除该目录重编。

## 插件错误

| 报错信息 | 原因 | 修复 |
|---------|------|------|
| Golden 函数不生效 | 未传 `--plugin` 参数 | 命令行加 `--plugin my_plugin.py` |
| Spec 未匹配 | TestSpec 类名/`__spec__` 注册名与 CSV 不一致 | 确保类名遵循 `PascalCase+TestSpec`（如 `AbsTestSpec`）或 `__spec__` dict 的 key 与 CSV `op_name`/`api_name` 一致 |
| 插件加载报错 | 语法或运行时错误 | 用 `python3 -c "import ast; ast.parse(open('my_plugin.py').read())"` 检查语法 |
| 同名注册 | 多个插件注册相同算子名 | 按搜索顺序先命中者生效（不覆盖、不打印警告，同文件内后者覆盖）；确保注册名称唯一 |
| TestSpec 加载失败 | `__spec__` 文件语法错或 Spec 类定义有误 | 检查 `__spec__` 标记与 Spec 类；语法错会记 error 日志（仅 `__spec__` 文件） |
