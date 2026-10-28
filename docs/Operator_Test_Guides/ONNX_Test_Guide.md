# ONNX算子测试指南

ONNX 通路是 E2E 模式（`python3 -m ttk e2e`）的 onnx 域：CSV 以 `onnx.<OpType>` 声明单算子，TTK 自动组装单算子 ONNX 模型，经 atc 离线编译为 .om 后在 NPU 上经 aclmdl 执行，覆盖「ONNX 模型解析 → atc ONNX 插件转换 → 离线模型装载执行」引擎层全链路；Golden 与被测共用同一份组模产物，由 CPU 侧 onnx ReferenceEvaluator 参照计算（同构参照）。

```
CSV (onnx.<OpType>)
   → 参数解析（onnx.defs schema，环境注册表即权威）
   → 输入生成（numpy 直通，无框架 tensor）
   → model_builder 组装单算子 ModelProto ──┬── atc 离线编译 .om → NPU aclmdl 执行（被测腿）
                                           └── onnx ReferenceEvaluator 执行（Golden 腿，同一模型字节）
   → 精度比对 / Dump / 结果列
```

与 E2E 的 torch/tf 域相比：无框架依赖（worker 进程仅预载 numpy + onnx），om 恒为静态图（eager ≡ 静态图），不支持动态 shape 与 aclgraph 模式。

---

# 环境准备

1、基本环境配置

- 建议 Python 3.8+，完成 CANN 包安装（onnx 通路使用其中 `atc` 离线编译器与 `libascendcl` 运行时）

```shell
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/cann/set_env.sh
# 指定路径安装
# source ${install_path}/cann/set_env.sh
```

2、Python 依赖

```shell
pip install onnx
```

> **版本约束**（两项都要满足）：
>
> - **onnx ≥ 1.13**：Golden 参照依赖 `onnx.reference.ReferenceEvaluator`；
> - **`onnx.defs.onnx_opset_version() ≤ 18`**：对齐 CANN atc ONNX 插件注册上限（超限在解析期报 `ONNX_OPSET_VERSION_UNSUPPORTED`）。CANN 9.2.x 配套 onnx 1.13.x。
>
> bfloat16 依赖 ml-dtypes（TTK `requirements.txt` 已含）。onnx 域无需安装 torch/tensorflow/torch_npu。

3、TTK工具安装

```shell
git clone https://gitcode.com/cann/ops-test-kit.git
cd ops-test-kit
pip install -r requirements.txt
```

4、检查环境

```shell
python3 -m ttk info
```

确认设备信息正常后再执行测试。

# 测试用例编写

CSV 使用 E2E 全量字段（23 个标准列），字段定义与 ONNX 域约定详见 [E2E 用例编写](./E2E_Case_Writing.md)。**参数来源是环境 onnx 包的算子 schema（`onnx.defs`），环境注册表即权威**——输入/属性名、类型、必选性均以环境注册表为准。

以 `examples/case_store/e2e/onnx_cases.csv` 为例：

```csv
testcase_name,api_name,tensor_view_shapes,tensor_dtypes,attributes,output_tensor_indexes,remark
onnx_add_f32,onnx.Add,"((2,3),(2,3))","('float32','float32')",,,基础双输入算子：f32 基线精度（输出槽声明缺省，返回值自动对账）
onnx_concat_i32,onnx.Concat,"(((2,3),(2,4)),None)","('int32','int32')",{'axis':1},"(1,)",Variadic 输入 + 必选属性 axis + 单输出槽声明
onnx_split_f32,onnx.Split,"((6,4),None,None,None)","('float32','float32','float32','float32')","{'axis':0,'num_outputs':3}","(1,2,3)",Optional 输入尾部省略 + 3 输出槽声明
```

## 关键约定

| 约定 | 说明 |
|------|------|
| api_name | `onnx.<OpType>`，不带 domain/opset 限定；带 domain 限定（如 `onnx.ml.*`）报 `ONNX_DOMAIN_UNSUPPORTED` |
| 输出槽 | 声明在 `tensor_view_shapes` 内（shape 填 `None`，下标进 `output_tensor_indexes`，dtype 声明进 `tensor_dtypes`），供对账 / dump / 精度判据锚定（低精度输出经 Golden Promote 抬 fp32 真值）；建模侧输出 shape 由 atc shape inference 推导 |
| Optional 输入 | 尾部省略语义：不声明即省略（少输入匹配按参数模型补 `None` 占位）；非输出位的 `None` 槽同为省略语义；省略仅限尾部，非尾部省略报 `ONNX_OPTIONAL_INPUT_UNSUPPORTED`（onnx parser 拒绝空串占位） |
| Variadic 输入 | 即 CSV 嵌套 TensorList（如 `((2,3),(2,4))`）；恒居尾部 |
| Variadic 输出 | 输出元数解析：`num_outputs` INT 属性（Split@18）优先，其次尾段 1-D 整型 sizes 输入（如 Split 的 `split`，其长度即输出数）；均无则建模期报错（数据依赖输出元数不支持） |
| 属性 | `attributes` 列即 ONNX 节点属性，一律按名传参；未知属性（校验期 `UNKNOWN_ATTRIBUTE`）与缺失必选属性（建模期报错）均显式失败，不静默忽略；含 TENSOR/GRAPH 载荷的属性类型（If/Loop 类控制流算子）报 `ONNX_ATTR_TYPE_UNSUPPORTED` |
| 布局列 | `tensor_formats` / `tensor_storage_shapes` / `tensor_view_strides` / `tensor_view_offsets` 不支持（ND 连续语义），填写报 `ONNX_LAYOUT_UNSUPPORTED` |
| dtype | 全量映射面见 `ttk/utilities/dtypes.py` 的 `ONNX_DTYPE_MAP`（float16/32/64、bfloat16、int8/16/32/64、uint8/16/32/64、bool、complex64/128、float8 族、int4 等）；无 TensorProto 映射的 dtype（uint1/int1、hifloat8/hifloat4、float4_e1m2、complex32、string）建模期报错；实际可执行面还取决于 atc 插件与 ReferenceEvaluator 的支持情况 |
| 输入 shape | 仅支持静态 shape（dim ≥ 0）；动态 dim（-1 槽位）校验期报 `ONNX_DYNAMIC_INPUT_UNSUPPORTED`（见下方失败码速查） |
| opset | 取环境 onnx 注册表最高版本，用例不指定 |

> 同一 CSV 内不得混用不同框架 API（torch / tf / onnx 框架检测会拒绝；onnx worker 进程预载 numpy + onnx，与 torch/tf 进程互斥）。

## 属性类型速查

`attributes` 列可构造的属性类型共 6 种（其余类型解析期报 `ONNX_ATTR_TYPE_UNSUPPORTED`）：

| schema 属性类型 | CSV 写法 | 示例 |
|----------------|---------|------|
| `INT` | 整数 | Concat 的 `axis`：`{'axis':1}`；Cast 的 `to`（**必选**，TensorProto 枚举值）：`{'to':1}`（float32） |
| `FLOAT` | 浮点 | Gemm 的 `alpha`/`beta`：`{'alpha':1.0,'beta':0.5}` |
| `STRING` | 字符串 | Conv 的 `auto_pad`：`{'auto_pad':'NOTSET'}` |
| `INTS` | 标量或整型序列 | Transpose 的 `perm`：`{'perm':(1,0,2)}`；Conv 的 `pads`/`strides`：`{'strides':(1,1)}` |
| `FLOATS` | 标量或浮点序列 | （按 schema 声明使用，写法同上） |
| `STRINGS` | 标量或字符串序列 | （按 schema 声明使用，写法同上） |

- `INTS`/`FLOATS`/`STRINGS` 接受**标量或序列**（标量自动抬为单元素序列），与 E2E 通用字段解析的宽容语义一致；
- 缺省属性不写入节点（走算子默认），必选属性缺失在建模期按算子/属性点名报错；
- `TENSOR(S)` / `SPARSE_TENSOR(S)` / `GRAPH(S)` / `TYPE_PROTO(S)` 载荷属性（If/Loop/Scan 类控制流算子）无法从 CSV 标量构造，解析期拒绝。

常用算子属性示例（以环境 onnx 1.13.x / opset 18 注册表为准）：

```csv
testcase_name,api_name,tensor_view_shapes,tensor_dtypes,attributes,output_tensor_indexes,remark
gemm_f32,onnx.Gemm,"((4,8),(6,8),None)","('float32','float32','float32')","{'alpha':1.0,'beta':0.5,'transB':1}","(2,)",标量属性 alpha/beta/transB（transB=1 时 B 形状按转置前声明；C 项省略，输出槽声明）
transpose_f32,onnx.Transpose,"((2,3,4),None)","('float32','float32')","{'perm':(2,0,1)}","(1,)",INTS 属性 perm（输出槽声明，shape 由 atc 推导）
relu_f16,onnx.Relu,"((8,16),None)","('float16','float16')",,"(1,)",无属性算子 + 输出槽声明（f16 Promote 锚点）
cast_i32,onnx.Cast,"((4,4),None)","('float32','int32')","{'to':6}","(1,)",必选属性 to=6（int32 的 TensorProto 枚举）
```

## 用环境 schema 查询算子参数

写用例前可用环境 onnx 包查询算子的输入/属性面（环境注册表即权威，与 TTK 解析同源）：

```shell
python3 -c "
import onnx.defs
s = onnx.defs.get_schema('Split')
for i in s.inputs:
    print('input :', i.name, i.option)
for o in s.outputs:
    print('output:', o.name, o.option)
for a in s.attributes.values():
    print('attr  :', a.name, a.type, '(required)' if a.required else '(optional)')
print('since :', s.since_version)
"
```

`option` 为 `Single`/`Optional`/`Variadic`——对应 CSV 的必选输入、尾部可省略输入（`None` 槽）与嵌套 TensorList；`required` 属性必须在 `attributes` 列按名提供。dtype 写法见 `ttk/utilities/dtypes.py` 的 `ONNX_DTYPE_MAP`。

# 精度测试

## 执行命令

```shell
# 基本执行（默认仅 eager 腿）
python3 -m ttk e2e -i onnx_cases.csv

# 指定用例
python3 -m ttk e2e -i onnx_cases.csv -t onnx_add_f32

# eager + 静态图双开：单次执行，EAGER 与 GRAPH CST 两列同值
python3 -m ttk e2e -i onnx_cases.csv --eager -c

# 输出结果
python3 -m ttk e2e -i onnx_cases.csv -o results.csv
```

## 执行流程

```
读取CSV → 生成输入数据（numpy） → 组装单算子 ONNX 模型 → atc 编译 .om → NPU aclmdl 执行
                                      └（同一模型字节）→ CPU ReferenceEvaluator 生成Golden → 精度比对 → 输出结果
```

atc 编译固定附加 `--precision_mode=must_keep_origin_dtype`（atc 默认 force_fp16 会静默降精度：f32 值超 65504 可得 inf 而 desc 仍报 float32）；soc_version 由 DSMI 探测的完整芯片名直传（如 Ascend910_9362）。

实现模块（`ttk/core_modules/framework_api/onnx/`，不依赖 torch/tf，numpy 直通）：

| 模块 | 职责 |
|------|------|
| `param_extractor.py` | `onnx.defs` schema（环境注册表）→ APIParamInfo；opset ≤ 18 与属性类型校验 |
| `model_builder.py` | 纯函数组装单算子 ModelProto（输入对位/属性归一/输出名生成）+ 进程级模型缓存 |
| `om_compiler.py` | atc 离线编译 + flock 编译缓存（缓存键含 atc ONNX 插件指纹） |
| `runner.py` | OmSession：om 装载/desc 查询/HBM 分配/执行轮（数据依赖输出在装载期拦截，报 `ONNX_DYNAMIC_OUTPUT_UNSUPPORTED`） |
| `runtime.py` | `libascendcl` 懒加载 CDLL 封装（aclmdl 装载/执行/拷贝） |

## 模式等价与限制

om 恒为静态图，因此 onnx 域 **eager ≡ 静态图**：

| 开关 | 行为 |
|------|------|
| 默认（仅 eager） | 单次执行，结果落 EAGER 列 |
| `--eager -c` | 双开去重：单次执行，(结果/性能/确定性) 镜像刷 EAGER 与 GRAPH CST 两列，两列恒同值（失败时错误文本同源镜像） |
| `--eager=false -c` | 单开静态图腿，等价 eager 执行 |
| `-d` | 快速失败：`ONNX_DYNAMIC_GRAPH_UNSUPPORTED` |
| `--aclgraph` | 快速失败：`ONNX_ACLGRAPH_UNSUPPORTED` |
| `--backend cpu` | 仅用于 Golden/输入生成；设备执行报 `ONNX_CPU_EXECUTION_UNSUPPORTED`（om 执行必须经 atc→aclmdl） |
| `--backend npusim` | 启动期快速失败：`ONNX_NPUSIM_UNSUPPORTED`（om 执行需真机 NPU——atc 编译 + aclmdl 装载执行，仿真不覆盖该链路） |
| `--golden-mode XPU` / `--xpu-perf` / `--compare cross_check` | 启动期快速失败：`ONNX_XPU_UNSUPPORTED`（三方远端按框架 dotted path 分发，无 onnx 通道；onnx 域 Golden 恒为 ReferenceEvaluator） |
| `--dump xpu` | 告警降级跳过（无远端 onnx 分发通道，无数据可落） |
| `--golden-mode NPU` | 语义退化同 `Enable`：onnx 域 Golden 恒为 CPU 侧 ReferenceEvaluator（同构参照，无独立 NPU golden），不产出 golden 性能列 |

## 结果列

- 精度结论落 `precision_status` / `eager_precision`，性能落 `eager_device_perf_us`（墙钟按轮平均）；
- `eager_kernel_count` / `eager_kernel_details` / `eager_cpu_perf_us` 为空（aclmdl 执行不经 torch dispatcher，无算子级明细，已知限制）；
- `--eager -c` 双开时单次执行，(精度/性能/确定性) 镜像刷 EAGER 与 GRAPH CST 两列，两列恒同值（失败时错误文本同源镜像）；`graph_dyn_*` / `graph_aclgraph_*` 列不产出。

## Golden 参照

- Golden 与被测共用同一 `model_builder` 产物（同构参照），由 `onnx.reference.ReferenceEvaluator` 在 CPU 侧计算；
- 低精度输出（fp16/bf16 等）经 Golden Promote 抬精度计算真值（输出槽 dtype 声明是 Promote 判据的锚点，务必声明）；
- `golden_api` 支持填另一个 `onnx.<OpType>` 做交叉参照，设为 `"disable"` 可禁用 Golden 生成（执行验证为主）；
- `--golden-mode NPU` 退化同 `Enable`（onnx 无独立 NPU golden 语义，恒为 CPU 侧 ReferenceEvaluator）；`--golden-mode XPU` 启动期拒绝（见模式等价与限制）；
- ReferenceEvaluator 不支持的算子/dtype 按 GOLDEN_FAILURE 占位，不拖累执行与性能列。

## 失败码速查

| 失败码 | 阶段 | 含义与处置 |
|--------|------|-----------|
| `ONNX_NPUSIM_UNSUPPORTED` | 启动 | `--backend npusim` + onnx 用例：om 执行需真机 NPU，仿真不覆盖；去掉 npusim 或改用 kernel/aclnn 模式 |
| `ONNX_XPU_UNSUPPORTED` | 启动 | `--golden-mode XPU` / `--xpu-perf` / `--compare cross_check` + onnx 用例：三方远端无 onnx 分发通道；去掉这些开关 |
| `UNKNOWN_ATTRIBUTE` | 用例校验 | `attributes` 列含 schema 未声明的属性名（拼写错误不静默忽略） |
| `ONNX_DOMAIN_UNSUPPORTED` | 解析 | api_name 带 domain 限定；仅支持默认 ai.onnx 域（`onnx.<OpType>`） |
| `ONNX_DYNAMIC_INPUT_UNSUPPORTED` | 用例校验 | `tensor_view_shapes` 输入槽含 dim<0（-1 动态标记）；onnx 域仅支持静态 shape（dim ≥ 0），动态 shape 覆盖请用 kernel/aclnn 模式 |
| `ONNX_ATTR_TYPE_UNSUPPORTED` | 解析 | 算子声明 TENSOR(S)/SPARSE_TENSOR(S)/GRAPH(S)/TYPE_PROTO(S) 载荷属性，CSV 标量属性无法构造 |
| `ONNX_OPSET_VERSION_UNSUPPORTED` | 解析 | onnx 包 opset 超 atc 插件上限（18）；换装配套 onnx 版本（CANN 9.2.x → onnx 1.13.x） |
| `ONNX_LAYOUT_UNSUPPORTED` | 绑定 | 填写了布局列（formats/storage_shapes/strides/offsets）；请清空这些列 |
| `ONNX_OPTIONAL_INPUT_UNSUPPORTED` | 建模 | 非尾部 Optional 输入省略；调整 `tensor_view_shapes` 声明 |
| `ONNX_DYNAMIC_OUTPUT_UNSUPPORTED` | om 装载 | 数据依赖 shape 的算子（NonZero/TopK 等）：atc 静默产出动态 dim om，输出 shape 无法恢复 |
| `ONNX_DYNAMIC_GRAPH_UNSUPPORTED` | 图模式 | `-d` 动态 shape 图模式不支持 |
| `ONNX_ACLGRAPH_UNSUPPORTED` | 图模式 | aclgraph 模式不支持 |
| `ONNX_CPU_EXECUTION_UNSUPPORTED` | 执行 | `--backend cpu` 下执行被测腿；CPU 仅用于 Golden/输入生成 |
| `ATC_NOT_FOUND` | 编译 | 未找到 atc；安装 CANN 并 source set_env.sh（或设 ASCEND_TOOLKIT_HOME） |
| `ATC_COMPILE_FAILED` | 编译 | atc 编译失败/超时（300s），stdout+stderr 尾部原文透传 |
| `OM_LOAD_FAILED` | 装载 | om 装载/desc 查询失败（含输出 dtype 无 TTK 映射） |
| `OM_EXECUTE_FAILED` | 执行 | aclmdlExecute 失败（ACL 错误码透传） |

> 建模期其余显式报错（按算子/参数点名）：必选属性缺失、必选输入缺失、Variadic 输入 0 张量、输入数超 schema 容量、dtype 无 TensorProto 映射、动态输入 dim、Variadic 输出元数不可解析、全 Optional 输出算子（GRU/RNN/LSTM）。

> 精度比对方法详见[精度比对方法](../Precision_Comparison.md)，Dump 调试详见[Dump 数据调试](../Dump_Debug.md)。

# 性能测试

```shell
# 默认执行（含性能采集，板端默认 3 次）
python3 -m ttk e2e -i onnx_cases.csv

# 设置执行次数
python3 -m ttk e2e -i onnx_cases.csv --run=5

# 关闭任务级Profiling采集（仍执行被测算子）
python3 -m ttk e2e -i onnx_cases.csv --task-prof=false
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--run` | 执行次数 | 板端3次 |
| `--warmup` | 预热（warmup 轮剔除建会话的一次性 setup：atc 编译命中后的 om 装载 + H2D） | 开启 |
| `--task-prof` | 任务级 Profiling 采集开关 | 开启 |

- 性能为**墙钟计时**：首轮（warmup）完成 om 装载、HBM 分配与 H2D 等全部重活，计时轮只走 `aclmdlExecute` + D2H；
- **kernel 明细为空**（已知限制）：aclmdl 执行不经 torch dispatcher，无法采集算子级 kernel 明细，性能数据以墙钟呈现。

# 缓存机制

| 缓存 | 键 | 位置/上限 | 失效 |
|------|----|----------|------|
| om 编译缓存 | md5(模型字节 + soc_version + atc 参数 + **atc ONNX 插件指纹**) | `<ttk 仓库根>/onnx_cache/<op_type>/<md5>/`（flock 并发安全） | atc ONNX 插件 .so 变更后（如 CANN 升级/重装）自动失效 |
| 会话缓存 | 结构键 md5(算子+输入形态+属性+soc+dev_id) → OmSession | 进程级 LRU，上限 4 个会话 | 同键不同输入值按内容指纹判定并自动重建；上限驱逐/替换时卸载旧会话释放 HBM |
| 模型缓存 | (算子, opset, 输入形态, 属性) → 序列化字节 | 进程级 LRU，上限 128 条 | 进程退出即失效 |

重复执行同签名用例（含 `--run` 多轮、双开镜像）只编译/装载一次。

# 多卡并行

```shell
# 使用全部可用NPU卡
python3 -m ttk e2e -i onnx_cases.csv

# 使用2张卡
python3 -m ttk e2e -i onnx_cases.csv --dev=2

# 每张卡2个进程
python3 -m ttk e2e -i onnx_cases.csv --pc=2

# 指定使用卡0
python3 -m ttk e2e -i onnx_cases.csv --devices=0
```

会话键含 dev_id 与 soc_version，跨卡执行互不串扰；om 编译缓存经 flock 并发安全，多进程同键只编译一次。

# 调试

```shell
# 调试单个用例（单用例独立日志目录自动生成，无需参数）
python3 -m ttk e2e -i onnx_cases.csv -t onnx_add_f32

# 固定随机种子（可复现）
python3 -m ttk e2e -i onnx_cases.csv --seed 42

# 仅校验CSV用例格式（不下设备）
python3 -m ttk e2e -i onnx_cases.csv --validate

# 精度失败自动dump
python3 -m ttk e2e -i onnx_cases.csv --dump-on-fail

# 离线数据两阶段（输入/Golden 制备与导入，onnx 域可用）
python3 -m ttk e2e -i onnx_cases.csv -t onnx_add_f32 --no-prof --dump in,golden --manual-data-dirs <dir>
python3 -m ttk e2e -i onnx_cases.csv -t onnx_add_f32 --manual-data-dirs <dir>
```

Dump 调试详见[Dump 数据调试](../Dump_Debug.md)，离线数据制备与导入详见[离线数据准备与导入](../Offline_Data_Prepare_and_Import.md)。

# 已知边界与限制

| 边界 | 说明 |
|------|------|
| Optional 输出省略 | atc 拒转声明了 Optional 输出的模型（MaxPool.Indices / Dropout.mask 等），建模只声明非 Optional 输出；全 Optional 输出算子（GRU/RNN/LSTM）建模期显式报错 |
| 数据依赖 shape 算子 | NonZero / TopK 等在 om 装载期报 `ONNX_DYNAMIC_OUTPUT_UNSUPPORTED`（atc 静默产出动态 dim om） |
| 动态 shape | 输入 dim 与 `-d` 图模式均不支持（atc 动态 shape 是非目标）：输入槽 dim<0 校验期报 `ONNX_DYNAMIC_INPUT_UNSUPPORTED`，`-d` 报 `ONNX_DYNAMIC_GRAPH_UNSUPPORTED` |
| 布局 | 仅 ND 连续语义，无 format/stride/offset 声明 |
| 控制流/子图算子 | 含 TENSOR/GRAPH 载荷属性的算子（If/Loop/Scan 等）解析期拒绝 |
| 三方 XPU 通路 | `--golden-mode XPU` / `--xpu-perf` / `--compare cross_check` 启动期拒绝（`ONNX_XPU_UNSUPPORTED`，含 manual_xpu_binaries 本地回放，首期边界）；`--dump xpu` 告警跳过 |
| 性能粒度 | 墙钟计时，kernel 明细为空 |
| 环境互斥 | onnx 用例与 torch/tf 用例不可混入同一 CSV（worker 进程预载互斥） |

# 通路选型

| 需求 | 推荐通路 |
|------|---------|
| 验证 atc ONNX 插件转换（模型导入链路） | onnx 域（唯一覆盖 atc→om 装载执行全链路的通路） |
| 无 torch/tensorflow 环境的引擎层单算子测试 | onnx 域（仅需 onnx 包 + CANN，worker 预载 numpy + onnx） |
| 算子级 kernel 明细 / 精确性能归因 | kernel / aclnn / geir（onnx 域仅墙钟计时，kernel 明细为空） |
| 动态 shape（`-1`）/ 数据依赖输出（NonZero/TopK 类） | aclnn / kernel（onnx 域静态 shape + 静态输出） |
| 框架 API 语义（dispatch/autograd 全链路） | E2E torch/tf 域 |

# 常用场景示例

```shell
# ONNX 域全量示例（6 例）
python3 -m ttk e2e -i examples/case_store/e2e/onnx_cases.csv

# eager + 静态图双开（单次执行，两列同值）
python3 -m ttk e2e -i examples/case_store/e2e/onnx_cases.csv --eager -c

# Variadic 输入/输出（Concat 嵌套 TensorList / Split 多输出）
python3 -m ttk e2e -i examples/case_store/e2e/onnx_cases.csv -t onnx_concat_i32,onnx_split_f32

# bf16 精度档（输出槽声明锚定 Golden Promote）
python3 -m ttk e2e -i examples/case_store/e2e/onnx_cases.csv -t onnx_matmul_bf16

# 重跑指定用例（E2E 不支持 --rerun，用 -t 按名圈选）
python3 -m ttk e2e -i onnx_cases.csv -t onnx_add_f32,onnx_matmul_bf16
```

> 通用参数（用例筛选/设备并行/精度控制/调试/结果输出）见[任务执行](../Task_Execution.md)
