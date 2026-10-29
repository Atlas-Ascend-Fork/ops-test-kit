# ACLNN 通路示例用例

单文件 `cases.csv` 覆盖 ACLNN 通路的主要功能面，每条用例的 `remark` 列即测试项；`golden.py` 为配套 golden 插件（`--plugin` 加载），补足 torch 同名映射不适配的 API。

```bash
A=examples/case_store/aclnn
# 全量（25 例）
python3 -m ttk aclnn -i $A/cases.csv --plugin $A/golden.py
```

## 基础精度链路（add / dtype 覆盖）

| 用例 | 测试项 |
|---|---|
| `add_fp16_broadcast` | fp16 标量广播，`scalar_dtypes` 随机标量 alpha |
| `add_bf16_bcast3d` | bfloat16 双侧广播（输出 rank=两输入维度并集） |
| `add_fp32_alpha_attr` | `attributes` 定值标量 alpha（`scalar_dtypes` 仍须声明，个数按接口对齐）+ `input_data_ranges` 必现值(0) |
| `add_fp64` / `add_int16` / `add_uint8` / `add_complex64` | dtype 补齐：float64 / int16 / uint8 / complex64 |
| `add_int8_negative` | int8 负值域定点比对（整型自动 binary_equal） |
| `add_noncontig` | 非连续输入：`tensor_view_strides`/`tensor_view_offsets`/`tensor_storage_shapes`（视图 (2,3) 取自存储 (3,6) 偏移 5） |

```bash
python3 -m ttk aclnn -i $A/cases.csv --plugin $A/golden.py \
    -t add_fp16_broadcast,add_bf16_bcast3d,add_fp32_alpha_attr,add_fp64,add_int16,add_uint8,add_complex64,add_int8_negative,add_noncontig
```

## TensorList 输入 / 输出

| 用例 | 测试项 |
|---|---|
| `cat_tensorlist_fp32` / `cat_bf16_dim0` / `cat_int8_dim_neg1` / `cat_int64_dim_neg1` | TensorList 变长输入（嵌套 shape）+ dim 正/负维 + dtype 补齐；`output_tensor_indexes` 显式声明 |
| `split_fp32` / `split_fp16_dim_neg1` / `split_int32_tail` | TensorList 输出 + `splitSections`（每份大小，尾块取余）+ `dim`；golden 由 golden.py 提供（torch.split 不收 `split_sections` 关键字） |

```bash
python3 -m ttk aclnn -i $A/cases.csv --plugin $A/golden.py \
    -t cat_tensorlist_fp32,cat_bf16_dim0,cat_int8_dim_neg1,cat_int64_dim_neg1,split_fp32,split_fp16_dim_neg1,split_int32_tail
```

## 标量参数与原地操作

| 用例 | 测试项 |
|---|---|
| `clamp_fp16_scalars` | 双标量参数（clipValueMin/Max）+ `scalar_data_ranges` 分段设取值范围；golden 由 golden.py 提供（torch.clamp 关键字为 min/max） |
| `inplace_fill_scalar_fp32` | 标量原地填充（`*Ref` 参数自动识别 `output_inplace_indexes`，无需手填） |
| `inplace_fill_tensor_int32` | 0-dim 张量原地填充 + int64 上限边界值域 |

```bash
python3 -m ttk aclnn -i $A/cases.csv --plugin $A/golden.py \
    -t clamp_fp16_scalars,inplace_fill_scalar_fp32,inplace_fill_tensor_int32
```

## 值约束输入（index_select）

| 用例 | 测试项 |
|---|---|
| `indexselect_fp32_range` | 索引张量值域约束：`input_data_ranges` 限 index∈[0,3]（随机默认域会越界）；golden 由 golden.py 提供（自动映射按「张量按位置、标量按名字」传参，`dim` 标量夹在 self 与 index 两个张量之间会错位，golden 需要按位置传参） |

```bash
python3 -m ttk aclnn -i $A/cases.csv --plugin $A/golden.py -t indexselect_fp32_range
```

## 动态输出 shape

| 用例 | 测试项 |
|---|---|
| `masked_select_elewise` / `masked_select_brc` | bool mask 同形/广播筛选，输出 N 由 mask 取值决定 |
| `nonzero_dynamic` | 输出 shape (N, ndim) 由输入取值决定 |
| `nonzero_v2_transposed` | 输出 shape (ndim, N) 转置布局；golden 由 golden.py 提供 |

```bash
python3 -m ttk aclnn -i $A/cases.csv --plugin $A/golden.py \
    -t masked_select_elewise,masked_select_brc,nonzero_dynamic,nonzero_v2_transposed
```

## 复杂属性与可选输入

| 用例 | 测试项 |
|---|---|
| `convolution_fp16` | 复杂属性（stride/padding/dilation/groups 等）+ 可选输入 bias 以 None 占位 + `tensor_formats` NCHW |

```bash
python3 -m ttk aclnn -i $A/cases.csv --plugin $A/golden.py -t convolution_fp16
```

dtype 覆盖：float16 / float32 / float64 / bfloat16 / int8 / int16 / int32 / int64 / uint8 / bool / complex64。

用例编写与命令行细节见 `.claude/skills/ttk-how-write-case/` 与 `ttk-how-run-test`；golden 编写见 `ttk-how-write-plugin`；判据详解见 `docs/reference/precision_comparison.md`。
