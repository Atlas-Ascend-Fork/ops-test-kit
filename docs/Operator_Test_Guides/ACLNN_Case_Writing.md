# ACLNN 用例编写

适用于 `python3 -m ttk aclnn`，使用 `TestcaseAclnn`，共 28 个字段（含[公共字段](../Test_Case_Generation.md#公共字段所有模式通用)与多卡 `my_rank`；不含 `attributes1`~`attributes9` 扩展列，全量 37 个）。

## 用例标识

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `api_name` | STRING | **是** | *(无)* | ACLNN API名称。如 `aclnnAdd`、`aclnnCat`。 |

## 张量属性

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `tensor_view_shapes` | SHAPELIKE_DYN_NESTED | **是** | `()` | 张量视图shape，支持TensorList嵌套。如 `"(((3,3),(3,2)),(3,5))"` 表示TensorList输入。 |
| `tensor_dtypes` | STRING_SCALAR_NESTED | **是** | `()` | 张量数据类型，支持嵌套。如 `"('float32','float32')"` 或 `"(('float32','float32'),'float32')"` |
| `tensor_formats` | STRING_SCALAR_NESTED | 否 | `('ND',)` | 张量格式，支持嵌套。如 `(('ND','ND'),'ND')` |
| `tensor_storage_shapes` | SHAPELIKE_STC_NESTED | 否 | `()` | 张量存储shape（用于非连续张量）。回退到视图shape。 |
| `tensor_view_offsets` | INT_CONTAINER_NESTED | 否 | `()` | 张量视图在存储中的偏移量。 |
| `tensor_view_strides` | SHAPELIKE_STC_NESTED | 否 | `()` | 张量视图步长。未指定时自动从shape计算。 |

## 输出属性

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `output_tensor_indexes` | INT_CONTAINER | 否 | *(自动)* | 指示哪些张量是输出的索引。未设置时按参数命名规则自动填充：以 `Ref`/`Out`/`Output`/`OutOptional`/`OutputOptional` 结尾或等于 `output` 的张量参数识别为输出；Backward/Grad API 会排除 `gradOutput`/`gradOut`/`grad_output`/`attentionOut` 以及非末位的 `output`；无匹配时回退到最后一个张量 |
| `output_inplace_indexes` | INT_CONTAINER | 否 | `()` | 原地操作输出索引。无需手动填写：框架自动从算子信息库推断（参数名以 `Ref` 结尾的张量参数即原地输出，输出覆写到该输入内存上） |

## 属性与标量

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `attributes` | DICT | 否 | `{}` | API属性参数（标量参数值）。如 `{'dim': -1}`、`{'value': 1.5}`。张量参数也可按形参名定值：随机生成后覆写对应张量槽（与 Kernel/E2E 模式一致）；声明 `manual_tensor_binaries` 时以文件内容为准 |
| `scalar_dtypes` | STRING_SCALAR_NESTED | 否 | `()` | 标量参数数据类型。支持ScalarList嵌套。顶层支持单元素压缩/广播（如 `('int64',)` 填充所有标量参数位）；ScalarList 的元素个数必须通过嵌套元组长度显式体现，不支持元素级压缩（如需 2 个元素须写 `(('int32','int32'),)`，`(('int32',),)` 只代表 1 个元素） |
| `scalar_data_ranges` | SHAPELIKE_FLOAT_SIGNED_NESTED | 否 | `((None, None),)` | 每个标量的数据范围 `(min, max)`。None 时 min=-2、max=2。标量值默认随机生成，也可在 `attributes` 中指定具体值（非必须） |

## Dump与手动数据

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `dump_file_prefix` | STRING | 否 | `None` | 数据dump文件的自定义文件名前缀。 |
| `manual_tensor_binaries` | FREE_EVAL | 否 | `()` | 手动张量二进制文件路径。作为Python表达式求值。 |
| `manual_golden_binaries` | STRING_CONTAINER | 否 | `()` | 手动Golden输出二进制文件路径。 |

## 非连续张量

通过 `tensor_view_strides`、`tensor_view_offsets`、`tensor_storage_shapes` 控制张量的非连续布局。视图 shape 和实际存储 shape 可以不同。未指定 strides/offsets/storage_shapes 时默认为连续张量。

| 字段组合 | 示例值 | 含义 |
|---------|--------|------|
| `tensor_view_shapes` | `"((2,3),)"` | 视图为 2x3 |
| `tensor_view_strides` | `"((4,1),)"` | 步长 (4,1)，行间 stride=4 |
| `tensor_view_offsets` | `"(5,)"` | 从存储第 5 个元素开始 |
| `tensor_storage_shapes` | `"((3,6),)"` | 存储 shape 为 3x6 |

## 通过 attributes 指定标量与张量值

在 `attributes` 字典中按接口定义的参数名指定标量参数的具体值（非必须，默认按 `scalar_data_ranges` 随机生成）；ScalarList 参数用列表值指定，如 `{'group_list': [1, 1, 1]}`。张量参数也可按形参名定值：随机生成输入后覆写对应张量槽，值按既有视图语义（strides/offsets）写入，TensorList 支持标量广播与逐子张量对位；声明 `manual_tensor_binaries` 时以文件内容为准，不走 attributes 覆写。

## 批一致性字段

`--deterministic-level 3` 控制确定性执行。
只有下列三字段齐全才构成 relation，单独配置 level 3 仍可正常执行。

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `batch_seed` | FREE_EVAL | `None` | `输入槽 -> 轴组 -> sample seed` |
| `batch_axis` | FREE_EVAL | `None` | `输入槽 -> 轴组 -> 逻辑轴`（视图 shape 上的轴） |
| `batch_slice_info` | FREE_EVAL | `None` | `输入槽 -> 轴组 -> sample 切片`，每个切片为 `(start, stop, step)` |

> **框架侧片段替换**（无需插件自实现）：三字段齐全时，输入生成期框架按
> `sample seed` 对声明切片逐样本重生成——同 seed + 同区段 shape 的样本跨用例逐位一致，
> 这是同 `batch_consistency_id` 组输出哈希比对成立的前提（整张量跨用例同 shape
> 同值播种需给定 `--seed`）。单槽声明的字段会按 TTK
> 单元素广播约定作用到全部输入槽；view/storage 布局错位（fp8 存储展平、非连续
> stride、非零 offset）的槽 warning 跳过。input 插件 kwargs 会收到
> `batch_applied: bool`（框架已替换标记），存量插件同 seed 重做替换幂等。

## 自定义输入插件（customize_inputs）

`--plugin` 算子 Spec 中可定义 `customize_inputs` 函数，在随机输入生成与 `attributes` 覆写之后、Golden 生成之前调用。形参按 aclnn 接口签名顺序传入（张量 + 标量 + 可选参数，含输出张量；张量为 torch.Tensor，非 torch 原生 dtype 时为 numpy 视图），kwargs 附带 `testcase_name`、`tensor_dtypes`、`input_ranges`、`batch_applied`（批一致性片段替换标记）、`testcase`（多卡 `my_rank`/`device_ids` 与槽表写入口）等上下文。

支持两种写法：

| 写法 | 插件行为 | 框架行为 |
|------|---------|---------|
| 原地 | 直接修改入参（`x.copy_(...)` / `x[...] = ...`），整体返回 `None` | 不做回写——入参与 dump/Golden 共享同一 storage，改动天然可见 |
| 返回值 | 返回新张量序列，与输入张量槽顺序一致（**纯输出槽除外**；TensorList 可嵌套可平铺，按展平顺序对位） | 逐槽 `view.copy_(val)` 拷入既有视图 |

返回值写法的槽位契约：

- 某槽返回 `None` = 该槽保持框架生成的原值；
- 回写是拷入既有视图而非替换对象：CSV 声明的 strides/offsets 布局与 `np_storages` 保持不变，dump/Golden/设备执行同源可见；
- 数量不符或 shape/dtype 不兼容会显式报错，不会静默丢弃；
- 动态/optional 槽（无生成张量）不能经返回值写，须用 `testcase` kwarg 的 `set_flat_tensor`/`set_flat_storage` 写入口。

```python
class AddTestSpec:
    def customize_inputs(x, y, alpha, out, **kwargs):
        # 形参按 aclnnAdd 接口签名顺序传入（含标量 alpha 与输出张量 out）
        return torch.full_like(x, 3.0), None
        # 返回值仅对应输入张量槽（x、y；纯输出槽 out 除外）：
        # 槽 0 回写 3.0；槽 1 为 None，保持框架生成的原值
```

> 若只需调整随机数据范围，可不写插件，直接在 CSV 设置 `input_data_ranges`。插件字段总表与各通路契约见 [`plugin/README.md`](../../ttk/core_modules/common/plugin/README.md)。

## 参考用例

`examples/case_store/aclnn/cases.csv`（单文件覆盖 dtype 与基本测试场景，`remark` 列即测试项；配套 `golden.py` golden 插件，`--plugin` 加载，详见目录内 README）：

| 用例 | 验证特性 | 关键列 |
|------|---------|--------|
| `add_fp16_broadcast` 等 add 系列 | 基础 API + dtype 覆盖（float16/32/64、bfloat16、int8/16、uint8、complex64；int32/int64 由 `cat_*`/`split_*`/`inplace_fill_*` 系列覆盖）+ 广播 + 标量参数 | `scalar_dtypes`、`attributes`（定值标量）、`input_data_ranges` |
| `add_noncontig` | 非连续张量 | `tensor_view_strides`、`tensor_view_offsets`、`tensor_storage_shapes` |
| `cat_*` / `split_*` | TensorList 输入（变长、嵌套 shape）/ TensorList 输出 | 嵌套 shape、`attributes`（dim/splitSections）、`output_tensor_indexes` |
| `clamp_fp16_scalars` | 双标量参数 + 标量定域 | `scalar_dtypes`、`scalar_data_ranges` |
| `inplace_fill_*` | 原地操作（`*Ref` 参数自动识别 `output_inplace_indexes`） | `input_data_ranges`、`precision_tolerances` |
| `indexselect_fp32_range` | 索引值域约束（随机默认域会越界） | `input_data_ranges` |
| `masked_select_*` / `nonzero_*` | bool mask 筛选 + 动态输出 | `input_data_ranges` |
| `convolution_fp16` | 复杂属性解析（stride/padding/dilation/groups）+ 可选输入 None 占位 + NCHW | `attributes`、`tensor_formats` |
