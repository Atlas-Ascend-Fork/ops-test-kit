# Kernel 用例编写

## 使用须知

Kernel 用例由[公共字段](../test_case_generation_guide.md)加上本文的专属字段组成。非必填列可省略，按默认值解析；输入/输出的数量和顺序由算子定义文件决定，须一一对应。

文档导览：

1. **[用例标识](#一用例标识)**
2. **[输入输出](#二输入输出)**
3. **[格式与回退](#三格式与回退)**
4. **[属性设置](#四属性设置)**
5. **[特殊选项](#五特殊选项)**
6. **[参考用例](#六参考用例)**

> [!NOTE]
> 算子的输入/输出/属性以算子定义（`xxx_def.cpp`）和算子信息库为准。信息库位于 `${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/config/{芯片系列号}/`，自定义算子为 `vendors/customize/...`。

## 一、用例标识

| 字段 | 是否必填 | 默认值 | 说明 |
|------|---------|--------|------|
| `op_name` | **是** | *(无)* | 算子名称，写算子信息库中的下划线格式名；信息库未收录时把驼峰名转下划线即可，如 `add`、`mat_mul_v3` |

## 二、输入输出

| 字段 | 是否必填 | 默认值 | 说明 |
|------|---------|--------|------|
| `input_shapes` | **是** | `()` | 输入张量 shape，支持 TensorList 嵌套。如 `"((128, 1024), (1, 1024))"` 表示两个张量；`"(((3,3),(3,2)),(3,5))"` 表示 TensorList(2) + 单个张量。可选输入不传时用 `None` 占位，对应 dtype/format 也写 `None` |
| `input_dtypes` | **是** | `()` | 输入 dtype，与 `input_shapes` 一一对应。支持广播：`('float32',)` 填充所有位置 |
| `output_shapes` | **是** | `None` | 输出张量 shape，支持 TensorList 嵌套 |
| `output_dtypes` | **是** | *(无)* | 输出 dtype，支持广播 |

- **TensorList**：算子信息库中该参数标为动态个数（DYNAMIC）时，即使只有 1 个张量也必须用嵌套格式。
- **动态 shape**：框架自动把正数维度替换为 `-1`，无需手工改写 `input_shapes`。

## 三、格式与回退

| 字段 | 是否必填 | 默认值 | 回退到 | 说明 |
|------|---------|--------|--------|------|
| `input_formats` | 否 | `('ND',)` | *(无)* | 输入张量格式，如 `('ND', 'NCHW')`。支持广播 |
| `input_ori_shapes` | 否 | → `input_shapes` | `input_shapes` | 原始输入 shape（格式转换前） |
| `input_ori_formats` | 否 | `('ND',)` | `input_formats` | 原始输入格式（用于格式转换）。支持广播 |
| `output_formats` | 否 | `('ND',)` | `output_ori_formats` | 输出张量格式。支持广播 |
| `output_ori_shapes` | 否 | `None` | `output_shapes` | 原始输出 shape |
| `output_ori_formats` | 否 | `('ND',)` | `output_formats` | 原始输出格式。支持广播 |

`output_formats` 与 `output_ori_formats` 互为回退：一方未填时取另一方，都未填时默认 `('ND',)`。

## 四、属性设置

`attributes` 保存算子属性，按参数名写值：

- **普通属性**：直接写，如 `{'transpose_x1': True}`。
- **ValueDepend 张量**（取值依赖输入数据的属性）：**必须**定值，如 ReduceMin 的 `{'axes': [0]}`。
- **值依赖其他输入的小张量**（索引、size 等，不宜随机）：也可定值，如 `{'group_list': [1, 1, 1]}`。

与输入同名的属性值，会覆盖该张量的随机生成。

## 五、特殊选项

### 1. 特殊属性设置

| 字段 | 是否必填 | 默认值 | 说明 |
|------|---------|--------|------|
| `output_inplace_indexes` | 否 | `()` | 原地输出索引。无需手动填写：框架从算子信息库自动推断 |
| `output_shape_unknown_indexes` | 否 | `()` | 编译期 shape 未知的输出索引，如 `"(0,)"` 表示第 0 个输出 |

### 2. 选项

| 字段 | 是否必填 | 默认值 | 说明 |
|------|---------|--------|------|
| `dump_file_prefix` | 否 | `None` | dump 目录的自定义名字（默认用例名） |
| `manual_input_binaries` | 否 | `()` | 手动输入二进制文件路径（Python 表达式）。支持 TensorList 嵌套 |
| `manual_golden_binaries` | 否 | `()` | 手动 Golden 二进制文件路径。支持 TensorList 嵌套 |
| `manual_xpu_binaries` | 否 | `()` | 已生成的 XPU 输出文件路径。`cross_check` 时直接使用本文件、不再调远端 XPU；须与手动输入、手动 Golden 成套配置 |

## 六、参考用例

`examples/case_store/kernel/cases.csv` 单文件覆盖各主要场景（每条用例的 `remark` 列即测试项）：

| 用例 | 验证特性 | 关键列 |
|------|---------|--------|
| `add_*` / `abs_*` | 基础精度链路：多维广播、值域定制与必现值 | `input_data_ranges`、`precision_tolerances` |
| `concat_d_tensorlist_in` | TensorList 动态输入个数 + 负 `concat_dim` 拼接 | `attributes`（concat_dim） |
| `split_tensorlist_out` | 拆分为 TensorList 输出 + `split_dim`/`num_split` | `attributes`，输出为多张量 |
| `reduce_min_valuedepend` | 规约 `axes`/`keep_dims` + ValueDepend 张量输入 | `attributes`（axes/keep_dims） |
| `non_zero_shape_unknown` | 输出 shape 由输入取值决定（数据相关动态输出） | `output_shape_unknown_indexes` |
| `zeros_like_constant` | 输出与输入同 shape 的零填充（无输入数据依赖） | `output_shapes` 跟随 `input_shapes` |
| `pad_float8` / `pad_float4` | 低比特 dtype：float8_e5m2 / float4_e2m1（通用算子 Pad） | `input_dtypes`；fp4 时尾轴 padFront 须偶数 |
| `quant_int8` / `requant_float8` | 量化比对演示：int8 默认 `binary_equal`，float8 建议 `--compare requant` | `attributes`（dst_type） |
| `matmul_v3_optional_inputs` | 矩乘可选输入（bias/offset_w 以 None 占位）+ `transpose_x1/x2` | `attributes`、可选输入 None 占位 |

示例覆盖 dtype：float16 / float32 / bfloat16 / int8 / int16 / int32 / int64 / complex64 / float8_e5m2 / float4_e2m1。

> [!IMPORTANT]
> 示例中 `concat_d`/`reduce_min`/`non_zero`/量化族无内置 golden；`split`/`pad` 的内置 golden 与真实语义有差异。以上用例均须由同目录 `golden.py` 插件提供 golden（`--plugin` 加载）。
