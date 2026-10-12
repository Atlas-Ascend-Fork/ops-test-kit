# E2E 通路示例用例

按框架分文件：`tf_cases.csv`（TensorFlow，13 例）与 `torch_cases.csv`（torch/torch_npu，24 例），每条用例的 `remark` 列即测试项。

```bash
E=examples/case_store/e2e
# TF 全量（13 例）；torch 全量（24 例）
python3 -m ttk e2e -i $E/tf_cases.csv
python3 -m ttk e2e -i $E/torch_cases.csv
```

> 同一表内不得混用 torch 与 tf API（框架检测会拒绝）；TF 进程预载 tensorflow、torch 进程预载 torch 系模块，两者互斥。受限内存容器（如 32GB cgroup）下 TF 用例建议 `--pc 2`：每个 TF worker 约 3GB host 内存，默认 `--pc 4` 可能触发 host 内存闸门等待。

## 基础精度链路（add / abs / relu）

| 用例 | 文件 | 测试项 |
|---|---|---|
| `add_f32_alpha` / `tf_add_f32` | torch / tf | 基础双输入 + 标量属性（alpha） |
| `add_f16` · `add_bf16` · `add_int32` / `tf_add_f16` · `tf_add_bf16` · `tf_add_int32` | torch / tf | dtype 覆盖：fp16 / bf16 / int32（整型 binary_equal） |
| `add_int8` · `add_int16` · `add_int64` · `add_uint8` · `add_bool` · `add_f64` · `add_complex64` | torch | dtype 补齐：int8 / int16 / int64 / uint8 / bool（binary_equal）、float64（mix）、complex64（isclose） |
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
| `tf_math_add` / `tf_matmul` | tf | tf.math / tf.linalg 命名空间；`tf_matmul` 为 f16 声明（NPU f16 matmul 原生计算。f32 变体经 GE 立方指令含 fp16 量化，默认 mix 判据数据相关不稳；如需 f32 用例须 `--compare close` 配 CSV 容差列——mix 判据按分档表 + Spec.tolerance 取值，不读 CSV 容差列） |

## TF 专属链路（历史缺陷回归）

| 用例 | 测试项 |
|---|---|
| `tf_scatter_div_stateful` | Resource* 有状态算子：mutable ref 槽自动 tf.Variable 化 + 更新值回读；bool 属性 use_locking 下发 |
| `tf_dequantize_scaled` | 量化 dtype quint8（numpy 存储别名 uint8）+ 字符串属性 `mode="SCALED"` 原值透传；0-D 标量输入（min_range/max_range） |
| `tf_range_valuedepend` | 0-D value-depend 输入：以 Python 值折叠 Const 供 GE infershape；输出 shape 数据相关 |

```bash
python3 -m ttk e2e -i $E/tf_cases.csv -t tf_scatter_div_stateful,tf_dequantize_scaled,tf_range_valuedepend
```

## torch_npu 与自定义算子

| 用例 | 测试项 |
|---|---|
| `npu_gelu_f16` | NPU 专属算子 + `golden_api` 交叉参照（CPU golden 用 `F.gelu`，实测 f16 最大偏差 ~6e-5）；CPU 环境预期 FAIL-fast |
| `causal_conv1d_fn_basic` | `torch.ops` 扩展命名空间 + 整型索引输入混排；`golden_api=disable`（自定义算子无 CPU 参照，执行验证为主）；aclnn 约束 dim∈[64,16384] |

```bash
python3 -m ttk e2e -i $E/torch_cases.csv -t npu_gelu_f16,causal_conv1d_fn_basic
```

dtype 覆盖：float16 / float32 / float64 / bfloat16 / int8 / int16 / int32 / int64 / uint8 / bool / complex64 / float8_e5m2 / float8_e4m3fn / float8_e8m0fnu（torch）、float16 / float32 / bfloat16 / int32 / quint8（tf）。fp4 / int4 为 torch 存储态 dtype（无计算内核），经 Kernel/GEIR 通路覆盖（`pad_float4` / `avgpool_update_int4`）。

用例编写与命令行细节见 `.claude/skills/ttk-how-write-case/` 与 `ttk-how-run-test`；E2E 字段定义见 [E2E 用例编写](../../../docs/Operator_Test_Guides/E2E_Case_Writing.md)。
