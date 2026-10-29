# E2E 用例编写

## 使用须知

`ttk e2e` 通过框架 API 端到端测试算子。用例由[公共字段](../test_case_generation_guide.md)加上本文字段组成，非必填列可省略。参数（张量数量/顺序/dtype、关键字参数）来自框架 API 的函数签名：torch / torch_npu 查官方文档，tf 查 TensorFlow 文档，onnx 查环境 onnx 包的算子 schema（`onnx.defs`）。

文档导览：

1. **[用例标识与张量](#一用例标识与张量)**
2. **[输出与属性](#二输出与属性)**
3. **[参考用例](#三参考用例)**

## 一、用例标识与张量

| 字段 | 是否必填 | 默认值 | 说明 |
|------|---------|--------|------|
| `api_name` | **是** | *(无)* | 框架 API 路径，如 `torch.add`、`torch.nn.functional.relu`、`torch.Tensor.relu_`；onnx写 `onnx.<OpType>`（不带 domain / opset），参数以环境 onnx 包的算子 schema 为准 |
| `tensor_view_shapes` | 否 | `()` | 输入张量 shape，支持 TensorList 嵌套。缺省时可由 Spec 的 `describe` 回填；无必选张量参数的 factory API（如 `torch.zeros`）可留空。onnx 中 shape 填 `None` 表示输出槽（对应 `output_tensor_indexes`）或省略的 Optional 输入（仅限尾部） |
| `tensor_dtypes` | 否 | `()` | 输入 dtype，支持嵌套与单值广播 |
| `tensor_formats` | 否 | `()` | 张量格式，支持广播 |
| `tensor_storage_shapes` | 否 | `()` | 非连续张量的存储 shape，缺省回退到视图 shape |
| `tensor_view_offsets` | 否 | `()` | 视图在存储中的偏移量 |
| `tensor_view_strides` | 否 | `()` | 视图步长，缺省自动计算 |

常用 API 形式：

| api_name 示例 | 说明 |
|---------------|------|
| `torch.add` | 模块函数，直接调用 |
| `torch.nn.functional.relu` | 子模块函数 |
| `torch.Tensor.relu_` | Tensor 方法（原地操作） |
| `torch.ops.cann_ops_transformer.causal_conv1d_fn` | torch extension 包注册的算子 |
| `onnx.Add` | `onnx.<OpType>` 声明单算子：TTK 组装模型 → atc 编译 .om → aclmdl 执行，与 onnx ReferenceEvaluator 比对 |

非连续张量示例：

| 字段 | 示例值 | 含义 |
|------|--------|------|
| `tensor_view_shapes` | `"((2,3),)"` | 视图为 2x3 |
| `tensor_view_strides` | `"((4,1),)"` | 步长 (4,1) |
| `tensor_view_offsets` | `"(5,)"` | 从存储第 5 个元素开始 |
| `tensor_storage_shapes` | `"((3,6),)"` | 存储 shape 为 3x6 |

onnx仅支持静态 shape（维度 ≥ 0），布局列（`tensor_formats` / `tensor_storage_shapes` / `tensor_view_strides` / `tensor_view_offsets`）不支持。

## 二、输出与属性

| 字段 | 是否必填 | 默认值 | 说明 |
|------|---------|--------|------|
| `output_tensor_indexes` | 否 | `()` | 输出张量索引。默认为空（输出从 API 返回值收集）；Inplace Tensor 方法（`torch.Tensor.xxx_`）自动填 `(0,)`；带 `out` 参数的 API 需手动指定 |
| `inplace_input_indexes` | 否 | `()` | 需要比对"被原地改写的输入"的张量索引（框架按此备份输入并参与比对） |
| `attributes` | 否 | `{}` | 框架 API 关键字参数，如 `{'alpha': 1.0}`。签名中必选的非张量参数**必须**提供。torch 用例可携带保留键 `CubeMathType`（int 0-4，不传入 API）：执行前设置 `torch_npu.npu.matmul/conv` 的立方计算精度模式（0 KEEP_DTYPE / 1 ALLOW_FP32_DOWN_PRECISION / 2 USE_FP16 / 3 USE_HF32 / 4 高精度档），用于 matmul/conv 类算子的精度模式验证；携带该键的用例先复位基线（KEEP_DTYPE + `allow_hf32=False`）再覆盖，仅 NPU + torch 生效。onnx即 ONNX 节点属性（按名传参，必选属性缺失报错） |
| `golden_api` | 否 | `""` | 替代 Golden 计算的 API，如 `torch.nn.functional.conv2d`；设为 `"disable"` 禁用 Golden |

### onnx 属性类型与 schema 查询

onnx的 `attributes` 列可构造的属性类型共 6 种：

| schema 属性类型 | CSV 写法 | 示例 |
|----------------|---------|------|
| `INT` | 整数 | Concat 的 `axis`：`{'axis':1}`；Cast 的 `to`（必选，TensorProto 枚举值）：`{'to':1}`（float32） |
| `FLOAT` | 浮点 | Gemm 的 `alpha`/`beta`：`{'alpha':1.0,'beta':0.5}` |
| `STRING` | 字符串 | Conv 的 `auto_pad`：`{'auto_pad':'NOTSET'}` |
| `INTS` / `FLOATS` / `STRINGS` | 标量或序列 | Transpose 的 `perm`：`{'perm':(1,0,2)}`；Conv 的 `pads`/`strides`：`{'strides':(1,1)}` |

`INTS`/`FLOATS`/`STRINGS` 接受标量或序列（标量自动按单元素序列处理）；未提供的属性走算子默认值，必选属性缺失会报错。

写用例前可用环境 onnx 包查询算子的输入/属性面（与 TTK 解析同源）：

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

`option` 为 `Single`/`Optional`/`Variadic`——对应 CSV 的必选输入、尾部可省略输入（`None` 槽）与嵌套 TensorList；`required` 属性必须在 `attributes` 列按名提供。

## 三、参考用例

`examples/case_store/e2e/` 按框架分文件：`tf_cases.csv` / `torch_cases.csv` / `onnx_cases.csv`（每条用例的 `remark` 列即测试项）。

### 1. tensorflow

| 用例 | 验证特性 | 关键列 |
|------|---------|--------|
| `tf_add_f32` 等 | 多 API（tf.raw_ops/nn/math/linalg/quantization/compat.v1）+ dtype 覆盖（fp16/bf16/int32/quint8） | `tensor_view_shapes`、`attributes` |
| `tf_matmul` | tf.linalg + f16 dtype（f32 变体在 NPU 经 GE 立方指令含 fp16 量化，如需 f32 用例须 `--compare close` 配 CSV 容差列） | `tensor_dtypes` |
| `tf_scatter_div_stateful` | Resource* 有状态算子（mutable ref 槽 tf.Variable 化 + 更新值回读） | `attributes`（use_locking） |
| `tf_dequantize_scaled` | 量化 dtype quint8 + 字符串属性 `mode="SCALED"` 原样传给算子 | `tensor_dtypes`、`attributes` |
| `tf_range_valuedepend` | 0-D value-depend 输入（Python 值折叠 Const 供 GE infershape）+ 数据相关输出 shape | `tensor_view_shapes`（0-D） |

### 2. torch / torch_npu

| 用例 | 验证特性 | 关键列 |
|------|---------|--------|
| `add_f32_alpha` | 基础双输入算子（f32 基线精度 + alpha 属性） | `attributes`（alpha） |
| `add_f16` / `add_bf16` / `add_int32` / `add_int8` / `add_int16` / `add_int64` / `add_uint8` / `add_bool` / `add_f64` / `add_complex64` | dtype 全覆盖：fp16 / bf16 / int32 / int8 / int16 / int64 / uint8 / bool（binary_equal）、float64（mix）、complex64（isclose） | `tensor_dtypes` |
| `pad_fp8_e5m2` / `pad_fp8_e4m3` / `pad_fp8_e8m0` | 低比特 fp8 族常量填充（e8m0 无 0 表示须填非零值）；fp4/int4 为 torch 存储态 dtype 不可执行，覆盖在 Kernel/GEIR 通路 | `tensor_dtypes`、`attributes`（pad/mode/value） |
| `add_range_mustappear` | 值域定制 + 第 3 元必现值 | `input_data_ranges` |
| `add_broadcast` | 广播语义（输入 shape 自动推导） | `tensor_view_shapes` |
| `add_strided_view` | 非连续视图（双侧 stride=2、offset 0/1） | `tensor_storage_shapes`、`tensor_view_strides`、`tensor_view_offsets` |
| `cat_tensorlist` | TensorList 嵌套输入（2 张量变长拼接） | `tensor_view_shapes`（嵌套） |
| `abs_01` | `torch.abs` 单输入、无属性 | — |
| `relu_f16` | `torch.nn.functional.relu` + 属性传递 + f16 低精度容差档 | `attributes`（inplace） |
| `relu_inp` | `torch.Tensor.relu_` 原地（自动 `output_tensor_indexes=(0,)`） | `output_tensor_indexes` |
| `add_out` | `out=` 输出参数位（CSV 声明输出槽） | `output_tensor_indexes` |
| `npu_gelu_f16` | NPU 专属算子 + `golden_api` 交叉参照（CPU golden 用 `F.gelu`；CPU 环境预期报错） | `golden_api` |
| `causal_conv1d_fn_basic` | `torch.ops` 自定义算子（torch-extension 包注册）+ 整型索引输入混排；`golden_api=disable` 执行验证 | `api_name`、`golden_api` |

### 3. onnx

| 用例 | 验证特性 | 关键列 |
|------|---------|--------|
| `onnx_add_f32` / `onnx_add_f16` / `onnx_add_broadcast` | 基础双输入 f32 基线 / f16 + 输出槽声明 / 广播（输出 shape 由 atc 推导） | `tensor_view_shapes`、`tensor_dtypes`、`output_tensor_indexes` |
| `onnx_concat_i32` / `onnx_split_f32` / `onnx_matmul_bf16` | Variadic 输入 + 必选属性 axis / Optional 输入尾部省略 + 3 输出槽声明（Variadic 输出）/ bf16 精度档 | `tensor_view_shapes`（嵌套/None 槽）、`attributes`、`output_tensor_indexes` |
