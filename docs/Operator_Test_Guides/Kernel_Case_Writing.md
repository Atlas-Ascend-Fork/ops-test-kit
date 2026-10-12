# Kernel 用例编写

适用于 `python3 -m ttk kernel` / `python3 -m ttk geir`，使用 `TestcaseKernel`。Kernel 标准 CSV 字段共 30 列（13 个[公共字段](../Test_Case_Generation.md#公共字段所有模式通用) + 17 个 Kernel 专属字段），另有 `attributes1`～`attributes9` 扩展属性列（全量 39 列）；可选列可按需省略，仓库示例 `examples/case_store/kernel/cases.csv` 仅保留 14 列基础子集，省略列按默认值解析。

GEIR 复用同一套字段并额外支持 `dyn_input_shapes` 列（标准字段 31 列，全量 40 列）。

具体shape通过 `input_shapes` / `output_shapes` 直接指定，动态shape由框架自动推导（将正数维度替换为 `-1`）。输入/输出的**数量和顺序**由算子定义文件决定，必须严格一一对应。TensorList 分组通过 shape 字段的嵌套结构表达，如 `(((3,3),(3,2)),(3,5))` 表示 TensorList(2) + 单个张量。**何时使用 TensorList 格式**：当算子信息库中 ParamType = DYNAMIC 时，即使只包含 1 个张量也必须用 TensorList 嵌套格式。

**算子定义与信息库来源**：
- **算子定义**：`xxx_def.cpp`（如 `add_def.cpp`），定义 INPUT/OUTPUT/ATTR 声明
- **算子信息库**：`aic-{芯片系列号}-ops-info-{算子仓[4:]}.json`，包含每个输入/输出的 name、支持 dtype、支持 format、paramType（optional/dynamic）、ValueDepend 等
  - builtin 算子信息库路径：`{ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/config/{芯片系列号}/aic-{芯片系列号}-ops-info-{算子仓[4:]}.json`
  - 自定义算子信息库路径：`{ASCEND_OPP_PATH}/vendors/customize/op_impl/ai_core/tbe/config/{芯片系列号}/aic-{芯片系列号}-ops-info.json`
  - 例如 add 算子在 `ascend910b` 芯片上的 builtin 信息库：`$ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/config/ascend910b/aic-ascend910b-ops-info-math.json`

## 用例标识

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `op_name` | STRING | **是** | *(无)* | 算子名称，优先使用算子信息库中 `opInterface.value` 的值（下划线格式）。若未配置 opInterface，则将驼峰算子名转为下划线格式即可。如 `add`、`mat_mul_v3`。 |

## 输入/输出Shape

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `input_shapes` | SHAPELIKE_STC_NESTED | **是** | `()` | 输入张量shape。支持TensorList嵌套。如 `"((128, 1024), (1, 1024))"` 表示两个张量，`"(((3,3),(3,2)),(3,5))"` 表示TensorList(2) + 单个张量。可选输入（paramType=optional）不传递时用 `None` 占位，对应的 dtype/format 也写 `None`。 |
| `input_dtypes` | STRING_SCALAR_NESTED | **是** | `()` | 输入数据类型。支持TensorList嵌套。支持广播：`('float32',)` 填充所有输入位置。如 `"('float32', 'float32')"` |
| `output_shapes` | SHAPELIKE_STC_EX_NESTED | **是** | `None` | 输出张量shape。支持TensorList嵌套。如 `"((128, 1024),)"` |
| `output_dtypes` | STRING_SCALAR_NESTED | **是** | *(无)* | 输出数据类型。支持TensorList嵌套。支持广播：`('float32',)` 填充所有位置。如 `"('float32',)"` |

## 格式与原始Shape

| 字段 | 类型 | 是否必填 | 默认值 | 回退到 | 说明 |
|------|------|---------|--------|--------|------|
| `input_formats` | STRING_SCALAR_NESTED | 否 | `('ND',)` | *(无)* | 输入张量格式。如 `('ND', 'NCHW')`。支持广播：`('ND',)` 填充所有位置 |
| `input_ori_shapes` | SHAPELIKE_STC_NESTED | 否 | → `input_shapes` | `input_shapes` | 原始输入shape（格式转换前）。 |
| `input_ori_formats` | STRING_SCALAR_NESTED | 否 | `('ND',)` | `input_formats` | 原始输入格式（用于格式转换）。支持广播 |
| `output_formats` | STRING_SCALAR_NESTED | 否 | `('ND',)` | `output_ori_formats` | 输出张量格式。支持广播 |
| `output_ori_shapes` | SHAPELIKE_STC_EX_NESTED | 否 | `None` | `output_shapes` | 原始输出shape。 |
| `output_ori_formats` | STRING_SCALAR_NESTED | 否 | `('ND',)` | `output_formats` | 原始输出格式。支持广播 |

## 属性

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `attributes` | DICT | 否 | `{}` | 算子属性（编译期和运行期）。如 `{'transpose_x1': True, 'transpose_x2': False}`。属性名匹配op_info输入名的会自动提取为特殊张量覆盖。当算子信息库中某个输入的 ValueDepend 为 OPTIONAL 或 REQUIRED 时，**必须**在 attributes 中指定该张量的值（如 ReduceMin 的 `axes`）。对于非 ValueDepend 但数值不能随机生成的小 shape 张量（如索引参数、shape 大小参数、size 参数等，其值与其他输入 shape 有关联），也可在 attributes 中指定，避免为小张量写输入定制函数（如 grouped_matmul 的 `group_list`）。 |

## 特殊属性

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `output_inplace_indexes` | INT_CONTAINER | 否 | `()` | 原地操作输出索引。无需手动填写：框架自动从算子信息库推断（当输出名称与输入名称相同时，表示输出覆写到该输入内存上） |
| `output_shape_unknown_indexes` | INT_CONTAINER | 否 | `()` | 编译期 shape 未知的**输出张量索引**。如 `"(0,)"` 表示第 0 个输出的 shape 在编译期无法确定 |

## 选项

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `dump_file_prefix` | STRING | 否 | `None` | 数据dump文件的自定义文件名前缀。 |
| `manual_input_binaries` | FREE_EVAL | 否 | `()` | 手动输入二进制文件路径。作为Python表达式求值。支持TensorList嵌套。 |
| `manual_golden_binaries` | FREE_EVAL | 否 | `()` | 手动Golden输出二进制文件路径。支持TensorList嵌套。 |
| `manual_xpu_binaries` | FREE_EVAL | 否 | `()` | 已生成的 XPU 输出文件路径。`cross_check` 时优先使用并跳过远端 XPU dispatch；必须与手动输入和手动 Golden 成套配置。 |

## 参考用例

`examples/case_store/kernel/cases.csv` 单文件覆盖各主要场景（每条用例的 `remark` 列即测试项，配套 `README.md` 含运行方式与预期结果）：

| 用例 | 验证特性 | 关键列 |
|------|---------|--------|
| `add_*` / `abs_*` | 基础精度链路：多维广播、值域定制与必现值 | `input_data_ranges`、`precision_tolerances` |
| `concat_d_tensorlist_in` | TensorList 动态输入个数 + 负 `concat_dim` 拼接 | `attributes`（concat_dim） |
| `split_tensorlist_out` | 拆分为 TensorList 输出 + `split_dim`/`num_split` | `attributes`，输出为多张量 |
| `reduce_min_valuedepend` | 规约 `axes`/`keep_dims` + ValueDepend 张量输入 | `attributes`（axes/keep_dims） |
| `non_zero_shape_unknown` | 输出 shape 由输入取值决定（数据相关动态输出） | `output_shape_unknown_indexes` 声明动态输出 |
| `zeros_like_constant` | 输出与输入同 shape 的零填充（无输入数据依赖） | `output_shapes` 跟随 `input_shapes` |
| `pad_float8` / `pad_float4` | 低比特 dtype：float8_e5m2 / float4_e2m1（通用算子 Pad） | `input_dtypes`；fp4 时尾轴 padFront 须偶数 |
| `quant_int8` / `requant_float8` | 量化判据路由演示：int8 输出默认 `binary_equal`（quant 判据需另在 TestSpec `tolerance` 声明 `{'int8': {'standard': 'quant'}}`）；float8 建议 CLI `--compare requant`（仅 hifloat8 自动路由，默认 `mixed` 亦可通过） | `attributes`（dst_type） |
| `matmul_v3_optional_inputs` | 矩乘可选输入（bias/offset_w 以 None 占位）+ `transpose_x1/x2` | `attributes`、可选输入 None 占位 |

dtype 覆盖 10 种：float16 / float32 / bfloat16 / int8 / int16 / int32 / int64 / complex64 / float8_e5m2 / float4_e2m1。其中 `concat_d`/`reduce_min`/`non_zero`/量化族无内置 golden；`split` 虽能回退 `numpy.split` 同名属性，但参数序与算子输入序对不上，均须由同目录 `golden.py` 插件提供（`--plugin` 加载）；`pad` 虽可回退 `numpy.pad` 内置 golden，仍建议由 `golden.py` 显式提供。
