# E2E 用例编写

适用于 `python3 -m ttk e2e`，使用 `TestcaseE2e`，共 23 个字段（含[公共字段](../Test_Case_Generation.md#公共字段所有模式通用)；不含 `attributes1`~`attributes9` 扩展列，全量 32 个）。

**参数来源**：E2E用例的参数（张量数量/顺序/dtype、关键字参数等）来自框架API的函数签名。torch / torch_npu 查看官方文档；tf 查看 TensorFlow 文档；onnx 查看环境 onnx 包的算子 schema（`onnx.defs`，环境注册表即权威）。

**API类型**：

| api_name 示例 | 类型 | 说明 |
|---------------|------|------|
| `torch.add` | 模块函数 | 直接调用 |
| `torch.nn.functional.relu` | 子模块函数 | 调用子模块中的函数 |
| `torch.Tensor.relu_` | Tensor方法 | 通过Tensor实例调用（原地操作） |
| `torch.ops.cann_ops_transformer.causal_conv1d_fn` | torch.ops 自定义算子 | torch extension 包注册的算子 |
| `onnx.Add` | ONNX 算子 | `onnx.<OpType>` 声明单算子（不带 opset 版本）：TTK 组模 → atc 编译 .om → NPU aclmdl 执行，与 onnx ReferenceEvaluator 比对 |

## 用例标识

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `api_name` | STRING | **是** | *(无)* | 框架API路径。如 `torch.add`、`torch.nn.functional.relu`、`torch.Tensor.relu_` |

## 张量属性

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `tensor_view_shapes` | SHAPELIKE_DYN_NESTED | 否 | `()` | 输入张量shape，支持TensorList嵌套。如 `"((2,3,4),(2,3,4))"`。缺省时可由 Spec 的 `describe` 按参数列组装回填（见[参数化用例](../Test_Case_Generation.md)）；无必选张量参数的 factory API（如 `torch.zeros`）可留空 |
| `tensor_dtypes` | STRING_SCALAR_NESTED | 否 | `()` | 输入张量数据类型。如 `"('float32','float32')"`。支持广播：`"('float32',)"` 填充所有位置。缺省时可由 `describe` 回填；factory API 可留空（同上） |
| `tensor_formats` | STRING_SCALAR_NESTED | 否 | `()` | 张量格式。支持广播 |
| `tensor_storage_shapes` | SHAPELIKE_STC_NESTED | 否 | `()` | 非连续张量的存储shape。回退到视图shape |
| `tensor_view_offsets` | INT_CONTAINER_NESTED | 否 | `()` | 张量视图在存储中的偏移量。 |
| `tensor_view_strides` | SHAPE_STRIDE | 否 | `()` | 张量视图步长。未指定时自动计算 |

## 输出与属性

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `output_tensor_indexes` | INT_CONTAINER | 否 | `()` | 输出张量索引。默认为空（输出从 API 返回值收集）；仅 Inplace Tensor 方法（`torch.Tensor.xxx_`）自动填充 `(0,)`；带 `out` 参数的API需手动指定输出位置索引 |
| `inplace_input_indexes` | INT_CONTAINER | 否 | `()` | 需要比对输入被原地改写的张量索引（框架按此备份输入并参与比对） |
| `attributes` | DICT | 否 | `{}` | 框架API关键字参数。如 `{'alpha': 1.0}`。API签名中的必选非张量参数**必须**提供。torch 用例可携带保留键 `CubeMathType`（int 0-4，不传入API）：执行前设置 `torch_npu.npu.matmul/conv` 立方计算精度模式（0 KEEP_DTYPE / 1 ALLOW_FP32_DOWN_PRECISION / 2 USE_FP16 / 3 USE_HF32 / 4 高精度档），用于 matmul/conv 类算子的精度模式场景验证；携带该键的用例先复位基线（KEEP_DTYPE + `allow_hf32=False`）再覆盖，未携带时保持 torch_npu 进程默认（同进程前序用例改写过则复位基线），仅 NPU + torch 框架生效 |
| `golden_api` | STRING | 否 | `""` | 替代Golden计算的API。如 `torch.nn.functional.conv2d`。设为 `"disable"` 可禁用Golden生成 |

## 批一致性字段

`--deterministic-level 3` 控制确定性执行。
只有下列三字段齐全才构成 relation，单独配置 level 3 仍可正常执行。

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `batch_seed` | FREE_EVAL | `None` | `输入槽 -> 轴组 -> sample seed` |
| `batch_axis` | FREE_EVAL | `None` | `输入槽 -> 轴组 -> 逻辑轴`（视图 shape 上的轴） |
| `batch_slice_info` | FREE_EVAL | `None` | `输入槽 -> 轴组 -> sample 切片`，每个切片为 `(start, stop, step)` |

> **框架侧片段替换**：三字段齐全时，输入生成期框架按 `sample seed`
> 逐样本重生成声明切片（同 seed 样本跨用例逐位一致；整张量跨用例同 shape
> 同值播种需给定 `--seed`）；npu_preprocess hook 的
> 元信息 kwargs 含 `batch_applied: bool` 标记。多卡（HCCL）用例不参与 batch
> 一致性分组比对（不产出 `batch_consistency_id`/`batch_summary` 比对结果）。

## 参考用例

`examples/case_store/e2e/` 目录下的示例（按框架分文件：`tf_cases.csv` / `torch_cases.csv` / `onnx_cases.csv`，按测试项分行，`remark` 列注明测试项；清单与运行方式见该目录 README）：

| 用例 | 框架 | 验证特性 | 关键列 |
|------|------|---------|--------|
| `tf_cases.csv` — `tf_add_f32` 等 | TensorFlow | 多 API（tf.raw_ops/nn/math/linalg/quantization/compat.v1）+ dtype 覆盖（fp16/bf16/int32/quint8） | `tensor_view_shapes`、`attributes` |
| `tf_cases.csv` — `tf_matmul` | TensorFlow | tf.linalg + f16 dtype（f32 变体在 NPU 经 GE 立方指令含 fp16 量化，如需 f32 用例须 `--compare close` 配 CSV 容差列） | `tensor_dtypes` |
| `tf_cases.csv` — `tf_scatter_div_stateful` | TensorFlow | Resource* 有状态算子（mutable ref 槽 tf.Variable 化 + 更新值回读） | `attributes`（use_locking） |
| `tf_cases.csv` — `tf_dequantize_scaled` | TensorFlow | 量化 dtype quint8 + 字符串属性 `mode="SCALED"` 原值透传 | `tensor_dtypes`、`attributes` |
| `tf_cases.csv` — `tf_range_valuedepend` | TensorFlow | 0-D value-depend 输入（Python 值折叠 Const 供 GE infershape）+ 数据相关输出 shape | `tensor_view_shapes`（0-D） |
| `torch_cases.csv` — `add_f32_alpha` | torch | 基础双输入算子（f32 基线精度 + alpha 属性） | `attributes`（alpha） |
| `torch_cases.csv` — `add_f16` / `add_bf16` / `add_int32` / `add_int8` / `add_int16` / `add_int64` / `add_uint8` / `add_bool` / `add_f64` / `add_complex64` | torch | dtype 全覆盖：fp16 / bf16 / int32 / int8 / int16 / int64 / uint8 / bool（binary_equal）、float64（mix）、complex64（isclose） | `tensor_dtypes` |
| `torch_cases.csv` — `pad_fp8_e5m2` / `pad_fp8_e4m3` / `pad_fp8_e8m0` | torch | 低比特 fp8 族常量填充（e8m0 无 0 表示须填非零值）；fp4/int4 为 torch 存储态 dtype 不可执行，覆盖在 Kernel/GEIR 通路 | `tensor_dtypes`、`attributes`（pad/mode/value） |
| `torch_cases.csv` — `add_range_mustappear` | torch | 值域定制 + 第 3 元必现值 | `input_data_ranges` |
| `torch_cases.csv` — `add_broadcast` | torch | 广播语义（输入 shape 自动推导） | `tensor_view_shapes` |
| `torch_cases.csv` — `add_strided_view` | torch | 非连续视图（双侧 stride=2、offset 0/1） | `tensor_storage_shapes`、`tensor_view_strides`、`tensor_view_offsets` |
| `torch_cases.csv` — `cat_tensorlist` | torch | TensorList 嵌套输入（2 张量变长拼接） | `tensor_view_shapes`（嵌套） |
| `torch_cases.csv` — `abs_01` | torch | `torch.abs` 单输入、无属性 | — |
| `torch_cases.csv` — `relu_f16` | torch | `torch.nn.functional.relu` + 属性传递 + f16 低精度容差档 | `attributes`（inplace） |
| `torch_cases.csv` — `relu_inp` | torch | `torch.Tensor.relu_` 原地（自动 `output_tensor_indexes=(0,)`） | `output_tensor_indexes` |
| `torch_cases.csv` — `add_out` | torch | `out=` 输出参数位（CSV 声明输出槽） | `output_tensor_indexes` |
| `torch_cases.csv` — `npu_gelu_f16` | torch_npu | NPU 专属算子 + `golden_api` 交叉参照（CPU golden 用 `F.gelu`；CPU 环境预期 FAIL-fast） | `golden_api` |
| `torch_cases.csv` — `causal_conv1d_fn_basic` | torch.ops | `torch.ops` 自定义算子（torch-extension 包注册）+ 整型索引输入混排；`golden_api=disable` 执行验证 | `api_name`、`golden_api` |
| `onnx_cases.csv` — `onnx_add_f32` / `onnx_add_f16` / `onnx_add_broadcast` | onnx | 基础双输入 f32 基线 / f16 + 输出槽声明（Promote 锚定）/ 广播（输出 shape 由 atc 推导） | `tensor_view_shapes`、`tensor_dtypes`、`output_tensor_indexes` |
| `onnx_cases.csv` — `onnx_concat_i32` / `onnx_split_f32` / `onnx_matmul_bf16` | onnx | Variadic 输入 + 必选属性 axis / Optional 输入尾部省略 + 3 输出槽声明（Variadic 输出）/ bf16 精度档 | `tensor_view_shapes`（嵌套/None 槽）、`attributes`、`output_tensor_indexes` |

## ONNX 域约定

执行链路、环境约束与失败码详见 [ONNX算子测试指南](./ONNX_Test_Guide.md)。

- **输出槽**：声明在 `tensor_view_shapes` 内（shape 填 `None`，下标进 `output_tensor_indexes`），供对账 / dump / 精度判据（`flat_output_dtypes`）锚定；建模侧输出 shape 由 atc shape inference 推导。非输出位的 `None` 槽 = 省略的 Optional 输入（尾部省略；非尾部省略建模期报错）；
- **属性**：`attributes` 列即 ONNX 节点属性（一律按名传参，必选属性缺失在建模期显式报错）；含 TENSOR/GRAPH 载荷的属性算子解析期拒绝（`ONNX_ATTR_TYPE_UNSUPPORTED`）；
- **布局列**（formats/storage_shapes/strides/offsets）不支持（ND 连续语义），填写报 `ONNX_LAYOUT_UNSUPPORTED`；
- **模式等价**：onnx 的 eager ≡ 静态图（om 恒为静态图），`--eager -c` 双开单次执行、EAGER 与 GRAPH CST 两列同值；`-d` / aclgraph 快速失败；
- **失败码**：`ATC_NOT_FOUND` / `ATC_COMPILE_FAILED`（atc 原文透传）/ `OM_LOAD_FAILED` / `OM_EXECUTE_FAILED` / `ONNX_DYNAMIC_OUTPUT_UNSUPPORTED`（数据依赖 shape 算子）/ `ONNX_DYNAMIC_GRAPH_UNSUPPORTED`（`-d`）/ `ONNX_ACLGRAPH_UNSUPPORTED`；
- **环境约束**：onnx 包须满足 `onnx.defs.onnx_opset_version() ≤ 18`（对齐 CANN atc 插件注册上限）；bf16 依赖 ml-dtypes。
