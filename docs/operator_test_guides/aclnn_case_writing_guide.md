# ACLNN 用例编写

## 使用须知

`ttk aclnn` 通过 aclnn\* C API 测试算子。用例由[公共字段](../test_case_generation_guide.md)加上本文字段组成；非必填列可省略，按默认值解析。张量、标量参数的数量、顺序与类型以接口头文件 `aclnnXxxGetWorkspaceSize` 的签名为准。

文档导览：

1. **[用例标识与张量](#一用例标识与张量)**
2. **[输出属性](#二输出属性)**
3. **[属性与标量](#三属性与标量)**
4. **[可选字段](#四可选字段)**
5. **[参考用例](#五参考用例)**

## 一、用例标识与张量

| 字段 | 是否必填 | 默认值 | 说明 |
|------|---------|--------|------|
| `api_name` | **是** | *(无)* | API 名，取头文件接口 `aclnnXxxGetWorkspaceSize` 的 `aclnnXxx` 部分，如 `aclnnAdd`、`aclnnCat` |
| `tensor_view_shapes` | 否 | `()` | 张量视图 shape，支持 TensorList 嵌套，如 `"(((3,3),(3,2)),(3,5))"`。缺省时可由 Spec 的 `describe` 按参数列回填 |
| `tensor_dtypes` | 否 | `()` | 张量 dtype，支持嵌套与单值广播 |
| `tensor_formats` | 否 | `('ND',)` | 张量格式，支持嵌套 |
| `tensor_storage_shapes` | 否 | `()` | 非连续张量的存储 shape，缺省回退到视图 shape |
| `tensor_view_offsets` | 否 | `()` | 视图在存储中的偏移量 |
| `tensor_view_strides` | 否 | `()` | 视图步长，缺省按 shape 自动计算 |

非连续张量示例：

| 字段 | 示例值 | 含义 |
|------|--------|------|
| `tensor_view_shapes` | `"((2,3),)"` | 视图为 2x3 |
| `tensor_view_strides` | `"((4,1),)"` | 步长 (4,1)，行间 stride=4 |
| `tensor_view_offsets` | `"(5,)"` | 从存储第 5 个元素开始 |
| `tensor_storage_shapes` | `"((3,6),)"` | 存储 shape 为 3x6 |

## 二、输出属性

| 字段 | 是否必填 | 默认值 | 说明 |
|------|---------|--------|------|
| `output_tensor_indexes` | 否 | *(自动)* | 输出张量索引。缺省按参数名自动识别（以 `Ref`/`Out`/`Output` 等结尾），无匹配时取末位张量 |
| `output_inplace_indexes` | 否 | `()` | 原地输出索引，`*Ref` 参数自动识别，无需手填 |

## 三、属性与标量

| 字段 | 是否必填 | 默认值 | 说明 |
|------|---------|--------|------|
| `attributes` | 否 | `{}` | API 属性参数，按接口形参名写值。如 `{'dim': -1}`。张量参数也可按形参名定值（随机生成后覆写对应张量槽）；声明 `manual_tensor_binaries` 时以文件内容为准 |
| `scalar_dtypes` | 否 | `()` | 标量参数 dtype，支持 ScalarList 嵌套与单元素广播（`('int64',)` 填充所有标量位） |
| `scalar_data_ranges` | 否 | `((None, None),)` | 标量数据范围 `(min, max)`，None 时 [-2, 2]。标量默认随机生成，也可在 `attributes` 中定值 |

> [!TIP]
> 只需调整随机数据范围时，不用写插件，直接设置 CSV 的 `input_data_ranges`。需要自定义输入生成时，在 Spec 中定义 `customize_inputs`，契约见[TestSpec 规范](../reference/test_spec.md)。

## 四、可选字段

| 字段 | 是否必填 | 默认值 | 说明 |
|------|---------|--------|------|
| `dump_file_prefix` | 否 | `None` | dump 目录的自定义名字（默认用例名） |
| `manual_tensor_binaries` | 否 | `()` | 手动张量二进制文件路径（Python 表达式） |
| `manual_golden_binaries` | 否 | `()` | 手动 Golden 二进制文件路径 |

## 五、参考用例

`examples/case_store/aclnn/cases.csv` 单文件覆盖 dtype 与基本测试场景（`remark` 列即测试项，配套 `golden.py` 插件，`--plugin` 加载）：

| 用例 | 验证特性 | 关键列 |
|------|---------|--------|
| `add_fp16_broadcast` 等 add 系列 | 基础 API + dtype 覆盖（float16/32/64、bfloat16、int8/16、uint8、complex64）+ 广播 + 标量参数 | `scalar_dtypes`、`attributes`、`input_data_ranges` |
| `add_noncontig` | 非连续张量 | `tensor_view_strides`、`tensor_view_offsets`、`tensor_storage_shapes` |
| `cat_*` / `split_*` | TensorList 输入（变长、嵌套 shape）/ TensorList 输出 | 嵌套 shape、`attributes`（dim/splitSections）、`output_tensor_indexes` |
| `clamp_fp16_scalars` | 双标量参数 + 标量取值范围 | `scalar_dtypes`、`scalar_data_ranges` |
| `inplace_fill_*` | 原地操作（`*Ref` 参数自动识别 `output_inplace_indexes`） | `input_data_ranges`、`precision_tolerances` |
| `indexselect_fp32_range` | 索引值域约束（随机默认域会越界） | `input_data_ranges` |
| `masked_select_*` / `nonzero_*` | bool mask 筛选 + 动态输出 | `input_data_ranges` |
| `convolution_fp16` | 复杂属性解析 + 可选输入 None 占位 + NCHW | `attributes`、`tensor_formats` |
