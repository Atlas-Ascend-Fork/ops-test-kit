# E2E 通路示例用例

按框架分文件：`torch_cases.csv`（torch/torch_npu，24 例）、`tf_cases.csv`（TensorFlow，13 例）与 `onnx_cases.csv`（ONNX，7 例），每条用例的 `remark` 列即测试项。同一 CSV 不得混用不同框架 API（onnx 与 torch/tf 的执行环境互斥，torch 与 tf 亦不能同一进程加载）。

> 较新 dtype/算子用例（fp8 族、`causal_conv1d_fn_basic`、`add_bool` 等）的通过依赖本地 CANN 版本的算子支持面，不支持时的预期错误签名见上级 [README 的 CANN 版本配套说明](../README.md)。

```bash
E=examples/case_store/e2e
```

## torch（torch_cases.csv）

CSV 以 `api_name` 声明被测 API（`torch.*` / `torch.nn.functional.*` / `torch.Tensor.*` 方法 / `torch_npu.*` / `torch.ops.*` 扩展命名空间）：TTK 按多源签名（aten schema / docstring / TypeError / inspect / pyi stub）解析出 ParamPlan 并生成输入，CPU 侧以 torch aten 同源实现生成 golden（`golden_api` 可改指其他基准实现），NPU 侧经 torch_npu 以 eager 或图模式执行后比对。

| 用例 | 测试项 |
|---|---|
| `add_f32_alpha` | 基础双输入 + 标量属性（alpha） |
| `add_f16` · `add_bf16` | dtype 覆盖：float16 / bfloat16 |
| `add_int32` · `add_int8` · `add_int16` · `add_int64` · `add_uint8` | dtype 覆盖：int32 / int8 / int16 / int64 / uint8（binary_equal） |
| `add_bool` | dtype 覆盖：bool（字节级比对）；部分版本线 NPU 侧按算术加实现（双 True 字节 0x02 而非 0x01）会判失败，属设备侧 bool 表示缺陷 |
| `add_f64` · `add_complex64` | dtype 覆盖：float64（mix）/ complex64（isclose） |
| `pad_fp8_e5m2` · `pad_fp8_e4m3` | 低比特 dtype：float8_e5m2 / float8_e4m3fn 常量填充（fp8 分档 mix） |
| `pad_fp8_e8m0` | float8_e8m0fnu 常量填充（e8m0 无 0 表示，须填非零值，binary_equal）；fp4/int4 为 torch 存储态 dtype（无内核，CPU/NPU 均不可执行，覆盖在 Kernel/GEIR 通路） |
| `add_range_mustappear` | `input_data_ranges` 值域定制 + 第 3 元必现值 0 |
| `add_broadcast` | 广播语义（输出 shape 自动推导） |
| `abs_01` · `relu_f16` | 单输入、无属性算子 / functional 公共 API + 属性传递 |
| `add_out` | `out=` 输出参数位（`output_tensor_indexes=(2,)` 声明输出槽） |
| `add_strided_view` | 非连续视图：双侧 stride=2、存储 (64,)、视图 (32,)、offset 0/1（`tensor_storage_shapes` / `tensor_view_strides` / `tensor_view_offsets`） |
| `cat_tensorlist` | TensorList 嵌套输入（2 张量变长拼接，经 `tensor_list_dist` 表达）+ dim 属性 |
| `relu_inp` | `torch.Tensor.relu_` 原地方法（自动 `output_tensor_indexes=(0,)`） |
| `npu_gelu_f16` | NPU 专属算子 + `golden_api` 指定 CPU 基准（CPU golden 用 `F.gelu`，实测 f16 最大偏差 ~6e-5） |
| `causal_conv1d_fn_basic` | `torch.ops` 扩展命名空间 + 整型索引输入混排；`golden_api=disable`（无 CPU 基准，只验证执行，结果 SKIP、比对列 SUPPRESSED）；aclnn 约束 dim∈[64,16384]；需部署配套 cann-ops-transformer 算子包，加载其 C++ 扩展依赖系统 `ninja` |

要点：

- **环境要求**：torch + torch_npu（主次版本须配套，按官方矩阵选择；不配套时 `import torch_npu` 直接崩溃，退出码 134）；fp8 族等较新 dtype/算子依赖本地 CANN 算子包支持面；
- **执行形态**：eager 默认开启；`-c` GE 静态图、`-d` 动态图、`--aclgraph`（`torch.compile` + npugraph_ex backend）三种图模式各自独立执行，结果分别写 EAGER / GRAPH CST / GRAPH DYN 列（onnx 的 eager≡静态图去重不适用于 torch）；`--fullgraph` 控制 torch.compile 全图；
- **golden 机制**：`golden_api` 指定 CPU 基准实现；`golden_api=disable` 表示无 CPU 基准、只验证执行；
- **输出位**：`output_tensor_indexes` 声明输出槽；`torch.Tensor` 原地方法（尾下划线）自动补 `(0,)`；
- **布局列**：支持 `tensor_storage_shapes` / `tensor_view_strides` / `tensor_view_offsets` 非连续视图；TensorList 经 `tensor_list_dist` 表达；
- **属性与值域**：`attributes` 列按参数名传递；`input_data_ranges` 定制输入值域；
- **性能**：经 msprof 采集（op 级 Avg 与 avg/max/min/calls 明细），`--task-prof=false` 时该列留空。

```bash
python3 -m ttk e2e -i $E/torch_cases.csv              # 仅 eager（默认）
python3 -m ttk e2e -i $E/torch_cases.csv --eager -c   # eager + 静态图
python3 -m ttk e2e -i $E/torch_cases.csv --backend npusim -t add_f32_alpha  # 无卡仿真（建议仅 eager）
```

## TF（tf_cases.csv）

CSV 以 `api_name` 声明被测 API（`tf.raw_ops.*` / `tf.math.*` / `tf.nn.*` / `tf.linalg.*` / `tf.quantization.*` 等）：TTK 按 TF 签名 + OpDef 元数据（is_ref / resource / 列表输入）解析出 ParamPlan 并生成输入（mutable ref 位转 tf.Variable），CPU 侧以 TF 原生实现生成 golden，NPU 侧经 npu_device 在 tf.function 内分发执行。

| 用例 | 测试项 |
|---|---|
| `tf_add_f32` | 基础双输入 + 标量属性（alpha） |
| `tf_add_f16` · `tf_add_bf16` | dtype 覆盖：float16 / bfloat16 |
| `tf_add_int32` | dtype 覆盖：int32（binary_equal） |
| `tf_add_broadcast` | 广播语义（输出 shape 自动推导） |
| `tf_abs` | 单输入、无属性算子 |
| `tf_relu_f32` · `tf_relu_f16` | `tf.nn` 公共 API + 属性传递 |
| `tf_math_add` | `tf.math` 命名空间 |
| `tf_matmul` | f16 声明（NPU f16 matmul 原生计算）；f32 变体经 GE 立方指令含 fp16 量化，默认 mix 比对可能数据相关不稳，如需 f32 用例须 `--compare close` 配 CSV 容差列（mix 的容差按分档表和 Spec.tolerance 取值，不读 CSV 容差列） |
| `tf_scatter_div_stateful` | Resource* 有状态算子：mutable ref 输入自动转 tf.Variable，并回读更新后的值；bool 属性 use_locking 传给算子 |
| `tf_dequantize_scaled` | 量化 dtype quint8（numpy 存储别名 uint8）+ 字符串属性 `mode="SCALED"` 原样传给算子；0-D 标量输入（min_range/max_range） |
| `tf_range_valuedepend` | 0-D value-depend 输入：先按 Python 值折叠为常量，供 GE 推导 shape；输出 shape 数据相关 |

要点：

- **环境要求**：tensorflow ≥2.12 + 昇腾 npu_device 插件；与 torch 进程内互斥（两大框架运行时冲突）；受限内存容器（如 32GB）建议 `--pc 2`（每个 worker 约 3GB 内存）；
- **图模式**：tf.function 原生图编译（无独立编译器后端）；`-c` 固定 input_signature（固定 shape）、`-d` 放开维度（None 维）；`--aclgraph` 不支持（跳过并告警）；
- **raw_ops 绑定**：`tf.raw_ops.*` 只收关键字参数，TTK 按 API 声明的参数名把位置输入绑成 kwargs；
- **mutable ref**：OpDef 声明 is_ref 的输入自动转 tf.Variable，执行后读回更新值（`tf_scatter_div_stateful`），各模式入口按 raw_inputs 复位初值；
- **const 折叠**：0-D value-depend 输入先按 Python 值折叠为 Const 节点，GE 才能推导输出 shape（`tf_range_valuedepend`）；
- **性能**：经 msprof 采集（`--task-prof=false` 时该列留空）。

```bash
python3 -m ttk e2e -i $E/tf_cases.csv --pc 2    # 受限内存容器（如 32GB）建议 --pc 2
python3 -m ttk e2e -i $E/tf_cases.csv --eager -c  # eager + 静态图
```

## ONNX（onnx_cases.csv）

CSV 以 `onnx.<OpType>` 声明单算子，可带显式 opset 段写成 `onnx.<opset>.<OpType>`（如 `onnx.13.Add`）：TTK 自动组装单算子 ONNX 模型 → atc 离线编译 .om → NPU 经 aclmdl 执行，与 CPU 侧 onnx ReferenceEvaluator 参考实现比对。参数（输入/属性）来自环境 onnx 包的算子 schema（`onnx.defs`）。

| 用例 | 测试项 |
|---|---|
| `onnx_add_f32` | 基础双输入 f32 基线（输出槽声明缺省，输出从返回值自动收集） |
| `onnx_add_f16` | dtype 覆盖 float16 + 输出槽声明 |
| `onnx_add_broadcast` | 广播语义：4×1×3 与 5×3 输入，输出 4×5×3 由 atc shape inference 推导 |
| `onnx_concat_i32` | Variadic 输入（TensorList 嵌套）+ 必选属性 axis + 单输出槽（int32 binary_equal） |
| `onnx_split_f32` | Optional 输入 split 尾部省略（省略的输入补 None 占位）；3 个 None 槽声明为输出槽（Variadic 输出 × num_outputs=3） |
| `onnx_matmul_bf16` | bf16 精度档 + 输出槽声明 |
| `onnx_add_v13` | 显式 opset 段（`onnx.13.Add`）：按声明版本建模，om 落 `onnx_om/onnx_add_v13/13/`；不带段时以环境 onnx 包最高 opset 建模 |

要点：

- **输出槽**声明在 `tensor_view_shapes` 内（shape 填 `None`，索引写入 `output_tensor_indexes`）；非输出位的 `None` 槽表示省略的 Optional 输入（仅限尾部）；
- **属性**即 ONNX 节点属性（`attributes` 列，一律按名传参）；opset 默认取环境 onnx 包的最高版本（要求不超过 18，即 atc 的标准算子注册上限），也可在 `api_name` 写显式段固定版本（如 `onnx.13.Add`，上限同为 18，超限解析期报错）；
- `tensor_formats` / `tensor_storage_shapes` / `tensor_view_strides` / `tensor_view_offsets` 布局列不支持（ND 连续语义），填写报 `ONNX_LAYOUT_UNSUPPORTED`；
- 模式等价：onnx 的 eager ≡ 静态图（om 恒为静态图），`--eager -c` 双开单次执行、两列同值；`-d` / aclgraph 直接报错（`ONNX_DYNAMIC_GRAPH_UNSUPPORTED` / `ONNX_ACLGRAPH_UNSUPPORTED`）；
- Optional 输出的算子（MaxPool.Indices / Dropout.mask / GRU 等）：atc 不接受带 Optional 输出的模型，构造模型时只声明非 Optional 输出（全 Optional 输出的算子构造模型时直接报错）；
- 数据依赖 shape 的算子（NonZero / TopK 等）在装载 om 时报 `ONNX_DYNAMIC_OUTPUT_UNSUPPORTED`；
- 环境要求：onnx ≥ 1.13（ReferenceEvaluator）、ml-dtypes（bf16）、CANN atc；性能经 msprof 采集（op 级 Avg 与 avg/max/min/calls 明细），`--task-prof=false` 时该列留空；om 编译缓存按用例归组在 `onnx_om/<用例名>/<opset>/model.om`。

```bash
python3 -m ttk e2e -i $E/onnx_cases.csv            # 仅 eager（默认）
python3 -m ttk e2e -i $E/onnx_cases.csv --eager -c # 双开：单次执行，EAGER/GRAPH CST 两列同值
```

通路指南（环境准备/执行链路/失败码/缓存机制）见 [E2E 算子测试指南](../../../docs/operator_test_guides/e2e_test_guide.md)。

dtype 覆盖：float16 / float32 / float64 / bfloat16 / int8 / int16 / int32 / int64 / uint8 / bool / complex64 / float8_e5m2 / float8_e4m3fn / float8_e8m0fnu（torch）、float16 / float32 / bfloat16 / int32 / quint8（tf）、float16 / float32 / bfloat16 / int32（onnx 为样例矩阵；TensorProto 全量 dtype 与 TTK 的映射见 `ttk/utilities/dtypes.py` 的 `ONNX_DTYPE_MAP`）。fp4 / int4 为 torch 存储态 dtype（无计算内核），经 Kernel/GEIR 通路覆盖（`pad_float4` / `avgpool_update_int4`）。

用例编写与命令行细节见 `.claude/skills/ttk-how-write-case/` 与 `ttk-how-run-test`；E2E 字段定义见 [E2E 用例编写](../../../docs/operator_test_guides/e2e_case_writing_guide.md)。
