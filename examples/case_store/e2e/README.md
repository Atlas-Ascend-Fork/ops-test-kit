# E2E 通路示例用例

按框架分文件：`tf_cases.csv`（TensorFlow，13 例）、`torch_cases.csv`（torch/torch_npu，24 例）与 `onnx_cases.csv`（ONNX，6 例），每条用例的 `remark` 列即测试项。

> 较新 dtype/算子用例（fp8 族、`causal_conv1d_fn_basic`、`add_bool` 等）的通过依赖本地 CANN 版本的算子支持面，不支持时的预期错误签名见上级 [README 的 CANN 版本配套说明](../README.md)。

```bash
E=examples/case_store/e2e
# TF 全量（13 例）；torch 全量（24 例）；onnx 全量（6 例）
python3 -m ttk e2e -i $E/tf_cases.csv
python3 -m ttk e2e -i $E/torch_cases.csv
python3 -m ttk e2e -i $E/onnx_cases.csv
```

> 同一 CSV 不得混用不同框架 API（onnx 与 torch/tf 的执行环境互斥）。受限内存容器（如 32GB）下 TF 建议 `--pc 2`（每个 worker 约 3GB 内存）。

## 基础精度链路（add / abs / relu）

| 用例 | 文件 | 测试项 |
|---|---|---|
| `add_f32_alpha` / `tf_add_f32` | torch / tf | 基础双输入 + 标量属性（alpha） |
| `add_f16` · `add_bf16` · `add_int32` / `tf_add_f16` · `tf_add_bf16` · `tf_add_int32` | torch / tf | dtype 覆盖：fp16 / bf16 / int32（整型 binary_equal） |
| `add_int8` · `add_int16` · `add_int64` · `add_uint8` · `add_bool` · `add_f64` · `add_complex64` | torch | dtype 补齐：int8 / int16 / int64 / uint8 / bool（binary_equal）、float64（mix）、complex64（isclose）；`add_bool` 为字节级比对——部分版本线 NPU 侧按算术加实现（双 True 字节 0x02 而非 0x01）会判失败，属设备侧 bool 表示缺陷，非用例问题 |
| `pad_fp8_e5m2` · `pad_fp8_e4m3` · `pad_fp8_e8m0` | torch | 低比特 dtype：float8_e5m2 / float8_e4m3fn / float8_e8m0fnu 常量填充（e5m2/e4m3 走 fp8 分档 mix，e8m0 无 0 表示须填非零值、binary_equal）；fp4/int4 为 torch 存储态 dtype（无 copy/pad 内核，CPU/NPU 均不可执行——其覆盖在 Kernel/GEIR 通路） |
| `add_range_mustappear` | torch | `input_data_ranges` 值域定制 + 第 3 元必现值 0 |
| `add_broadcast` / `tf_add_broadcast` | torch / tf | 广播语义（输出 shape 自动推导） |
| `abs_01` / `tf_abs` | torch / tf | 单输入、无属性算子 |
| `relu_f16` / `tf_relu_f32` · `tf_relu_f16` | torch / tf | functional / tf.nn 公共 API + 属性传递 |

## 结构特性

| 用例 | 文件 | 测试项 |
|---|---|---|
| `add_out` | torch | `out=` 输出参数位（`output_tensor_indexes=(2,)` 声明输出槽） |
| `add_strided_view` | torch | 非连续视图：双侧 stride=2、存储 (64,)、视图 (32,)、offset 0/1（`tensor_storage_shapes`/`tensor_view_strides`/`tensor_view_offsets`） |
| `cat_tensorlist` | torch | TensorList 嵌套输入（2 张量变长拼接）+ dim 属性 |
| `relu_inp` | torch | `torch.Tensor.relu_` 原地方法（自动 `output_tensor_indexes=(0,)`） |
| `tf_math_add` / `tf_matmul` | tf | tf.math / tf.linalg 命名空间；`tf_matmul` 为 f16 声明（NPU f16 matmul 原生计算。f32 变体经 GE 立方指令含 fp16 量化，默认 mix 比对结果可能数据相关不稳；如需 f32 用例须 `--compare close` 配 CSV 容差列——mix 的容差按分档表和 Spec.tolerance 取值，不读 CSV 容差列） |

## TF 专属链路（历史缺陷回归）

| 用例 | 测试项 |
|---|---|
| `tf_scatter_div_stateful` | Resource* 有状态算子：mutable ref 输入自动转 tf.Variable，并回读更新后的值；bool 属性 use_locking 传给算子 |
| `tf_dequantize_scaled` | 量化 dtype quint8（numpy 存储别名 uint8）+ 字符串属性 `mode="SCALED"` 原样传给算子；0-D 标量输入（min_range/max_range） |
| `tf_range_valuedepend` | 0-D value-depend 输入：先按 Python 值折叠为常量，供 GE 推导 shape；输出 shape 数据相关 |

```bash
python3 -m ttk e2e -i $E/tf_cases.csv -t tf_scatter_div_stateful,tf_dequantize_scaled,tf_range_valuedepend
```

## torch_npu 与自定义算子

| 用例 | 测试项 |
|---|---|
| `npu_gelu_f16` | NPU 专属算子 + `golden_api` 交叉参照（CPU golden 用 `F.gelu`，实测 f16 最大偏差 ~6e-5）；CPU 环境预期报错 |
| `causal_conv1d_fn_basic` | `torch.ops` 扩展命名空间 + 整型索引输入混排；`golden_api=disable`（自定义算子无 CPU 参照，执行验证为主）；aclnn 约束 dim∈[64,16384]。需部署配套 cann-ops-transformer 算子包，且加载其 C++ 扩展依赖系统 `ninja`（缺失时报 `Ninja is required to load C++ extensions`） |

```bash
python3 -m ttk e2e -i $E/torch_cases.csv -t npu_gelu_f16,causal_conv1d_fn_basic
```

## ONNX（onnx_cases.csv）

CSV 以 `onnx.<OpType>` 声明单算子：TTK 自动组装单算子 ONNX 模型 → atc 离线编译 .om → NPU 经 aclmdl 执行，与 CPU 侧 onnx ReferenceEvaluator 参考实现比对。参数（输入/属性）来自环境 onnx 包的算子 schema（`onnx.defs`）。

| 用例 | 测试项 |
|---|---|
| `onnx_add_f32` | 基础双输入 f32 基线（输出槽声明缺省，输出从返回值自动收集） |
| `onnx_add_f16` | dtype 覆盖 float16 + 输出槽声明 |
| `onnx_add_broadcast` | 广播语义：4×1×3 与 5×3 输入，输出 4×5×3 由 atc shape inference 推导 |
| `onnx_concat_i32` | Variadic 输入（TensorList 嵌套）+ 必选属性 axis + 单输出槽（int32 binary_equal） |
| `onnx_split_f32` | Optional 输入 split 尾部省略（省略的输入补 None 占位）；3 个 None 槽声明为输出槽（Variadic 输出 × num_outputs=3） |
| `onnx_matmul_bf16` | bf16 精度档 + 输出槽声明 |

要点：

- **输出槽**声明在 `tensor_view_shapes` 内（shape 填 `None`，索引写入 `output_tensor_indexes`）；非输出位的 `None` 槽表示省略的 Optional 输入（仅限尾部）；
- **属性**即 ONNX 节点属性（`attributes` 列，一律按名传参）；opset 取环境 onnx 包支持的最高版本（要求 `onnx.defs.onnx_opset_version() ≤ 18`，即 CANN atc 支持的版本上限）；
- `tensor_formats` / `tensor_storage_shapes` / `tensor_view_strides` / `tensor_view_offsets` 布局列不支持（ND 连续语义），填写报 `ONNX_LAYOUT_UNSUPPORTED`；
- 模式等价：onnx 的 eager ≡ 静态图（om 恒为静态图），`--eager -c` 双开单次执行、两列同值；`-d` / aclgraph 直接报错（`ONNX_DYNAMIC_GRAPH_UNSUPPORTED` / `ONNX_ACLGRAPH_UNSUPPORTED`）；
- Optional 输出的算子（MaxPool.Indices / Dropout.mask / GRU 等）：atc 不接受带 Optional 输出的模型，构造模型时只声明非 Optional 输出（全 Optional 输出的算子构造模型时直接报错）；
- 数据依赖 shape 的算子（NonZero / TopK 等）在装载 om 时报 `ONNX_DYNAMIC_OUTPUT_UNSUPPORTED`；
- 环境要求：onnx ≥ 1.13（ReferenceEvaluator）、ml-dtypes（bf16）、CANN atc；perf 为墙上时钟计时（kernel 明细为空，已知限制）。

```bash
python3 -m ttk e2e -i $E/onnx_cases.csv            # 仅 eager（默认）
python3 -m ttk e2e -i $E/onnx_cases.csv --eager -c # 双开：单次执行，EAGER/GRAPH CST 两列同值
```

通路指南（环境准备/执行链路/失败码/缓存机制）见 [E2E 算子测试指南](../../../docs/operator_test_guides/e2e_test_guide.md)。

dtype 覆盖：float16 / float32 / float64 / bfloat16 / int8 / int16 / int32 / int64 / uint8 / bool / complex64 / float8_e5m2 / float8_e4m3fn / float8_e8m0fnu（torch）、float16 / float32 / bfloat16 / int32 / quint8（tf）、float16 / float32 / bfloat16 / int32（onnx 为样例矩阵；TensorProto 全量 dtype 与 TTK 的映射见 `ttk/utilities/dtypes.py` 的 `ONNX_DTYPE_MAP`）。fp4 / int4 为 torch 存储态 dtype（无计算内核），经 Kernel/GEIR 通路覆盖（`pad_float4` / `avgpool_update_int4`）。

用例编写与命令行细节见 `.claude/skills/ttk-how-write-case/` 与 `ttk-how-run-test`；E2E 字段定义见 [E2E 用例编写](../../../docs/operator_test_guides/e2e_case_writing_guide.md)。
