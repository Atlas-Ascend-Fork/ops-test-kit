# FAQ 一本通



## 问题定位自查

1. TTK版本是否最新，可执行 `python3 -m ttk -v` 查看版本号
2. 自查命令是否有误，可执行 `python3 -m ttk kernel --help` 查看参数说明
3. CANN包是否已安装并source环境变量
4. 如果出现算子内部报错，请检查算子名称书写是否有误
5. 如果使用自定义插件，检查注册器名称是否与CSV中的算子名匹配

## 环境问题

### 运行报错 "ASCEND_HOME_PATH not found"

TTK依赖CANN工具包。请先安装CANN并source环境变量：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### CANN分包环境变量设置

如使用CANN分包安装，需额外设置环境变量：

```shell
source /usr/local/Ascend/latest/bin/setenv.bash
export ASCEND_HOME_PATH=/usr/local/Ascend/latest
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/latest
```

常见报错（未source环境变量）：
- `ImportError: libhccl.so`
- `AttributeError: 'NoneType' object has no attribute 'acl_init'`

### Python版本问题

建议Python 3.8+，可通过以下命令确认版本：

```shell
python3 --version
```

### `import torch_npu` 崩溃（退出码 134）报 `Duplicated key 'pinned_reserve_segment_size_mb'`？

torch 与 torch_npu 版本不配套：报错栈看不出是版本问题，实为两者底层库注册冲突。按 torch_npu 官方配套矩阵重装配套版本组合即可恢复（实测组合：`torch==2.12.0` + `torch-npu==2.12.0.post2`，CANN 9.3.0）。

## 用例编写问题

### CSV中数据类型怎么写？

支持的数据类型字符串：`float16`/`fp16`、`float32`/`fp32`、`float64`/`fp64`、`bfloat16`/`bf16`、`int8`、`int16`、`int32`、`int64`、`uint8`、`bool`、`complex64`、`complex128` 等。

### CSV中包含特殊字符的字段需要用双引号包裹吗？

是的。CSV中包含括号、逗号、引号的字段（如shape、dtype元组、字典等）需要用双引号包裹：

```csv
"((128, 1024), (1, 1024))"     # 正确
((128, 1024), (1, 1024))       # 错误，逗号会被解析为CSV分隔符
```

### 输出Shape依赖输入Tensor中的值，无法从输入shape推导（如NonZero）怎么写CSV？

`output_shapes` 填写一个占位shape（非空），同时在 `output_shape_unknown_indexes` 中标记未知输出的索引：

```csv
output_shapes,"((8, 1716),)"
output_shape_unknown_indexes,"(0,)"
```

框架会在运行时确定实际输出shape。

### TensorList算子怎么写CSV？

通过 `input_shapes` 字段的嵌套结构标记TensorList。多个tensor放在同一层元组中表示它们组成一个TensorList：

```csv
input_shapes,"(((3,3),(3,2),(3,4)),(3,5))"
```

表示前3个输入tensor组成一个TensorList，第4个是单独张量。

### 如何查看CSV中定义了哪些用例？

使用 `list` 命令预览：

```shell
python3 -m ttk list -i cases.csv
python3 -m ttk list -i cases.csv --op add
```

> [!NOTE]
> `ttk list` 仅列出用例名，并要求 `testcase_name` 列存在且非空（否则直接报错），不做其他字段校验；完整校验请用对应模式 `--validate`。

## 执行问题

### 算子太大执行超时或超出内存

- 设置单用例超时时间：`--proc-timeout=300`
- 设置HBM内存限制：`-l 10`（10GB，仅 Kernel/ACLNN 通路注册）
- 预留HBM内存：`--reserve-hbm=512`（仅 Kernel 通路注册）

### 多卡并行时部分用例失败

- 检查设备是否正常：`python3 -m ttk info`
- 排除问题设备：`--devices='^2,^3'`（每个被剔除 ID 均需带 `^` 前缀，等价于全选后剔除，也可写 `'0-7,^2,^3'`）
- 减少每张卡进程数：`--pc=1`

### 如何复现某次测试结果？

固定随机种子：

```shell
python3 -m ttk kernel -i cases.csv --seed 42
```

### 如何快速验证环境是否正常？

```shell
# 查看设备信息
python3 -m ttk info

# 运行内置示例
python3 -m ttk kernel -i examples/case_store/kernel/cases.csv -t add_fp16_broadcast
```

## ONNX 通路问题

### onnx 用例报 `ONNX_OPSET_VERSION_UNSUPPORTED`？

环境 onnx 包的 opset 版本超过 CANN atc ONNX 插件注册上限（18）。TTK 要求 onnx ≥ 1.13（Golden 依赖 ReferenceEvaluator）且 `onnx.defs.onnx_opset_version() ≤ 18`——CANN 9.2.x 配套 onnx 1.13.x：

```shell
python3 -c "import onnx.defs; print(onnx.defs.onnx_opset_version())"   # 须 <= 18
pip install "onnx==1.13.1"
```

### onnx 用例能用 `--backend cpu` 跑吗？

不能。`--backend cpu` 下执行被测腿报 `ONNX_CPU_EXECUTION_UNSUPPORTED`——om 执行必须经 NPU（atc 编译 .om → aclmdl 执行），CPU 侧仅用于 Golden（ReferenceEvaluator）与输入生成。

### `-d` 或 `--aclgraph` 报 `ONNX_DYNAMIC_GRAPH_UNSUPPORTED` / `ONNX_ACLGRAPH_UNSUPPORTED`？

onnx om 恒为静态图：eager ≡ 静态图。去掉这两个开关即可；需要双列结果时用 `--eager -c`（单次执行，EAGER 与 GRAPH CST 两列同值）。

### NonZero / TopK 等算子报 `ONNX_DYNAMIC_OUTPUT_UNSUPPORTED`？

这类算子的输出 shape 依赖输入数据，atc 生成的 om 会带动态维，输出 shape 无法恢复。onnx 通路不支持此类算子，请改用 kernel/aclnn 模式覆盖。

### 怀疑 atc 编译缓存脏了怎么办？

onnx om 编译缓存位于 ttk 仓库根目录的 `onnx_cache/<op_type>/` 下。缓存按 atc 插件版本区分，CANN 升级/重装后自动失效；仍怀疑缓存脏时删除该目录重编即可：

```shell
rm -rf onnx_cache/<op_type>/
```

### onnx 用例与 torch/tf 用例能混在同一个 CSV 吗？

不能。同一 CSV 内不得混用不同框架 API，框架会直接拒绝（onnx 与 torch/tf 的执行环境互斥）。按框架分文件存放，示例见 `examples/case_store/e2e/`（`torch_cases.csv` / `tf_cases.csv` / `onnx_cases.csv`）。

### onnx失败码速查

| 失败码 | 阶段 | 含义与处置 |
|--------|------|-----------|
| `ONNX_NPUSIM_UNSUPPORTED` | 启动 | `--backend npusim` + onnx 用例：om 执行需真机 NPU，仿真不覆盖 |
| `ONNX_XPU_UNSUPPORTED` | 启动 | `--golden-mode XPU` / `--xpu-perf` / `--compare cross_check` + onnx 用例：三方远端无 onnx 分发通道 |
| `UNKNOWN_ATTRIBUTE` | 用例校验 | `attributes` 列含算子未声明的属性名（拼写错误会直接报错） |
| `ONNX_DOMAIN_UNSUPPORTED` | 解析 | api_name 带 domain 限定；仅支持默认 ai.onnx（`onnx.<OpType>`） |
| `ONNX_DYNAMIC_INPUT_UNSUPPORTED` | 用例校验 | `tensor_view_shapes` 输入槽含 dim < 0（-1 动态标记）；onnx仅支持静态 shape |
| `ONNX_ATTR_TYPE_UNSUPPORTED` | 解析 | 算子声明 TENSOR(S)/SPARSE_TENSOR(S)/GRAPH(S)/TYPE_PROTO(S) 载荷属性，CSV 标量属性无法构造 |
| `ONNX_OPSET_VERSION_UNSUPPORTED` | 解析 | onnx 包 opset 超 atc 插件上限（18）；换装配套 onnx 版本（CANN 9.2.x → onnx 1.13.x） |
| `ONNX_LAYOUT_UNSUPPORTED` | 绑定 | 填写了布局列（formats/storage_shapes/strides/offsets）；请清空这些列 |
| `ONNX_OPTIONAL_INPUT_UNSUPPORTED` | 建模 | 非尾部 Optional 输入省略；调整 `tensor_view_shapes` 声明 |
| `ONNX_DYNAMIC_OUTPUT_UNSUPPORTED` | om 装载 | 数据依赖 shape 的算子（NonZero/TopK 等）：atc 生成的 om 带动态维，输出 shape 无法恢复 |
| `ONNX_DYNAMIC_GRAPH_UNSUPPORTED` | 图模式 | `-d` 动态 shape 图模式不支持 |
| `ONNX_ACLGRAPH_UNSUPPORTED` | 图模式 | aclgraph 模式不支持 |
| `ONNX_CPU_EXECUTION_UNSUPPORTED` | 执行 | `--backend cpu` 下执行被测腿；CPU 仅用于 Golden/输入生成 |
| `ATC_NOT_FOUND` | 编译 | 未找到 atc；安装 CANN 并 source set_env.sh（或设 ASCEND_TOOLKIT_HOME） |
| `ATC_COMPILE_FAILED` | 编译 | atc 编译失败或超时（300 秒），报错内容为 atc 输出尾部的原文 |
| `OM_LOAD_FAILED` | 装载 | om 装载/desc 查询失败（含输出 dtype 无 TTK 映射） |
| `OM_EXECUTE_FAILED` | 执行 | 算子执行失败，ACL 错误码照原样显示 |

### onnx 编译与执行缓存

| 缓存 | 说明 | 失效条件 |
|------|------|---------|
| om 编译缓存 | 编译结果缓存在 `<ttk 仓库根>/onnx_cache/<op_type>/` 下，缓存键由模型内容、SoC 版本、atc 参数等决定 | atc 插件更新（如 CANN 升级/重装）后自动失效 |
| 进程内会话缓存 | 同「算子 + 输入形态 + 属性 + 设备」的编译模型复用，最多保留 4 个 | 输入内容变化时自动重建；超出上限释放最久未用的会话、释放 HBM |
| 进程内模型缓存 | 按「算子、opset、输入形态、属性」复用已构建的模型定义，最多 128 条 | 进程退出即失效 |

同签名的用例重复执行（含 `--run` 多轮、`--eager -c` 同时开启）只编译/装载一次。

## 离线数据问题

| 报错 | 检查项 |
| --- | --- |
| `manual-data case ... was not found` | CSV 的完整 `testcase_name` 与搜索根目录 |
| `slot count ... != CSV` | 文件漏传 / 索引不连续 / CSV 变化 |
| `filename dtype ... != CSV storage dtype` | dump/load 的 dtype |
| `saved shape ... != device output shape` | 设备输出 shape 回归 / 旧 CSV |
| `byte size ... != expected` | 文件截断 / dtype、shape 变化 |
| `unexpected file in manual-data case` | 目录中无关文件（日志/JSON/子目录） |
| kernel 离线落盘/回放需相同编译模式 | 两次命令用相同 `-d`/`-c`/`-b` |

## 精度问题

### 精度比对失败如何排查？

1. 使用 `--dump-on-fail` 在失败时自动Dump数据
2. 使用 `--dump in,out,golden --dump-format npy` 保存输入、输出和Golden到npy文件
3. 用 `-t case_name` 锁定到单个用例（单用例执行时框架自动产出该用例的独立日志目录）
4. 检查CSV中 `precision_tolerances` 和 `absolute_precision` 设置是否合理（这两个旧字段只在部分比对方法下生效，保留兼容；默认 `mixed` 的容差在 TestSpec `tolerance` 中调整）

```shell
# 失败时自动Dump数据
python3 -m ttk kernel -i cases.csv --dump-on-fail

# Dump输入和Golden到npy文件
python3 -m ttk kernel -i cases.csv --dump in,golden --dump-format npy
```

### 精度比对方法选择建议

| 场景 | 推荐方法 | 参数 |
|------|---------|------|
| 浮点运算常规测试（默认） | 混合容差（生态算子开源精度标准） | `--compare mixed`（默认） |
| 统计误差均值/最大值 | 统计相对误差（社区标准） | `--compare stat_rel_err` |
| 逐点 isclose | 数值近似 | `--compare close` |
| 大规模向量整体趋势 | 余弦相似度 | `--compare cosine` |
| 整型运算/需要精确结果 | 二进制精确 | `--compare binary` |
| hifloat8类型 | 重量化 | `--compare requant`（自动） |
| 三方交叉校验 | 三方交叉校验 | `--compare cross_check`（需 `third_party`） |

### 如何调整精度容差？

在CSV中通过以下字段设置：

```csv
precision_tolerances,"((0.001, 0.001),)"
absolute_precision,1e-8
```

- `precision_tolerances`：每个输出的 (rtol, ptol) 对
- `absolute_precision`：绝对精度容差（atol）。单个浮点数应用于全部输出；tuple/list 按输出索引逐输出配置，如双输出 `"(1e-8, 1e-5)"`

> [!IMPORTANT]
> 这两个 CSV 字段是旧版字段（保留兼容），`precision_tolerances` 只会被 `--compare close`/`cosine`/`requant` 及 TestSpec 声明的 `quant` 读取；`absolute_precision` 仅 `close` 读取。默认的 `mixed` 不读 CSV 容差字段，需在 TestSpec 的 `tolerance` 中覆盖（需 `--plugin`）：
>
> ```python
> tolerance = {"float32": {"standard": "mix_tolerance", "rtol": 0.002, "atol": 1e-5}}
> ```
>
> 可覆盖项：`rtol` / `atol` / `required_matched_ratio` / `max_abs_error_limit`（配置后不再按值域自动计算上限，完全以配置值为准，见[精度比对方法](../reference/precision_comparison.md)）。

## 插件问题

### 自定义Golden函数不生效

1. 确认已通过 `--plugin` 参数传入插件文件路径
2. 确认 TestSpec 类名/`__spec__` 注册名与CSV中的 `op_name` 一致（类名遵循 `PascalCase+TestSpec`，如 `AbsTestSpec`）
3. 检查插件文件是否有语法错误

### 多个插件有同名算子注册

> [!IMPORTANT]
> TestSpec（`__spec__`）注册同名时，先加载的插件目录优先；旧的 `__golden__`/`__input__` 注册同名时，后扫描的插件覆盖先扫描的。两种情况都不会告警，请确保注册名称唯一。
