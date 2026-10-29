# GEIR 用例编写

## 使用须知

`ttk geir` 把算子组装为 GE 计算图整体编译执行。用例字段与 Kernel 模式相同，额外增加 `dyn_input_shapes` 列控制图编译时的动态维度；非必填列可省略，按默认值解析。

文档导览：

1. **[用例标识与输入](#一用例标识与输入)**
2. **[格式与属性](#二格式与属性)**
3. **[可选字段](#三可选字段)**
4. **[动态维度](#四动态维度)**
5. **[参考用例](#五参考用例)**

## 一、用例标识与输入

| 字段 | 是否必填 | 默认值 | 说明 |
|------|---------|--------|------|
| `op_name` | **是** | *(无)* | 算子名（下划线格式），如 `add`、`mat_mul_v3` |
| `input_shapes` | **是** | `()` | 输入 shape，支持 TensorList 嵌套；可选输入用 `None` 占位 |
| `input_dtypes` | **是** | `()` | 输入 dtype，支持广播：`('float32',)` 填充所有位置 |
| `output_shapes` | **是** | `None` | 输出 shape，支持 TensorList 嵌套 |
| `output_dtypes` | **是** | *(无)* | 输出 dtype，支持广播 |

## 二、格式与属性

| 字段 | 是否必填 | 默认值 | 回退到 | 说明 |
|------|---------|--------|--------|------|
| `input_formats` | 否 | `('ND',)` | *(无)* | 输入格式，如 `('ND', 'NCHW')` |
| `input_ori_shapes` | 否 | → `input_shapes` | `input_shapes` | 原始输入 shape（格式转换前） |
| `input_ori_formats` | 否 | `('ND',)` | `input_formats` | 原始输入格式 |
| `output_formats` | 否 | `('ND',)` | `output_ori_formats` | 输出格式 |
| `output_ori_shapes` | 否 | `None` | `output_shapes` | 原始输出 shape |
| `output_ori_formats` | 否 | `('ND',)` | `output_formats` | 原始输出格式 |

`output_formats` 与 `output_ori_formats` 互为回退：一方未填时取另一方，都未填时默认 `('ND',)`。

`attributes` 保存算子属性，按参数名写值：普通属性直接写；ValueDepend 张量（信息库中标为 OPTIONAL / REQUIRED 的输入）**必须**定值；值依赖其他输入的小张量（索引、size 等）也可定值。与输入同名的属性值，会覆盖该张量的随机生成。

## 三、可选字段

| 字段 | 是否必填 | 默认值 | 说明 |
|------|---------|--------|------|
| `output_inplace_indexes` | 否 | `()` | 原地输出索引，框架从算子信息库自动推断 |
| `output_shape_unknown_indexes` | 否 | `()` | 编译期 shape 未知的输出索引，如 `"(0,)"` |
| `dump_file_prefix` | 否 | `None` | dump 文件的自定义名前缀 |
| `manual_input_binaries` | 否 | `()` | 手动输入二进制文件路径（Python 表达式） |
| `manual_golden_binaries` | 否 | `()` | 手动 Golden 二进制文件路径 |
| `manual_xpu_binaries` | 否 | `()` | 已生成的 XPU 输出文件路径，须与手动输入、手动 Golden 成套配置 |

## 四、动态维度

动态编译（`-d`）时，`dyn_input_shapes` 描述图编译使用的 shape：`-1` 表示未知维度，`-2` 表示未知秩。未指定时，所有维度按 `-1` 处理（与 Kernel 仅替换正数维度不同）。

| 字段 | 作用 | 示例 |
|------|------|------|
| `input_shapes` | 实际执行的数据 shape | `"((128, 1024), (1, 1024))"` |
| `dyn_input_shapes` | 图编译时使用的 shape 描述 | `"((128, -1), (1, 1024))"` |

上例中第 1 个输入第 0 维固定 128、第 1 维动态，第 2 个输入固定。单输入未知秩可写 `"((-2,),)"`。

> [!IMPORTANT]
> GEIR 默认静态编译（`-c`），`dyn_input_shapes` 仅在 `-d` 下生效，需显式指定 `-d`。

## 五、参考用例

> [!TIP]
> 现有 Kernel 用例 CSV 可直接用于 GEIR，无需改动；需要精确控制某输入维度的动态性时，在末尾追加 `dyn_input_shapes` 列即可。

`examples/case_store/geir/cases.csv` 覆盖 GEIR 主要场景（`remark` 列即测试项，配套 `golden.py` 插件）：

### 1. dyn_input_shapes 语义

| 用例 | 验证特性 | 关键列 |
|------|---------|--------|
| `add_dyn_default` | 列缺省：动态编译时全维度自动替换为 `-1` | `dyn_input_shapes` |
| `add_dyn_tail_fixed` | 尾维固定：拼接维动态、特征维编译期固定 | `dyn_input_shapes` |
| `add_dyn_mixed_bcast` | 混合固定/动态维度 + 广播输入整体固定 | `dyn_input_shapes` |
| `add_dyn_rank_unknown` | 未知秩 desc（`(-2,)`），算子须 dynamicRankSupport | `dyn_input_shapes` |

### 2. 可选输入与复杂属性

| 用例 | 验证特性 | 关键列 |
|------|---------|--------|
| `conv2d_nchw_optional` | NCHW + 复杂属性（strides/pads/dilations/groups）+ bias/offset_w 可选输入 `None` + 数据维动态、filter 固定 | `attributes`、`dyn_input_shapes` |
| `arg_max_v2_range_pin` | ValueDepend 张量（dimension）经 `input_data_ranges` 定值 | `input_data_ranges` |
| `arg_max_v2_dtype_attr` | Type 型属性（`ATTR(dtype, Type)`）字符串值按枚举整数值填写 | `attributes` |
| `smoothl1_loss_inf_attr` | 非有限浮点属性（`sigma=inf`） | `attributes` |

### 3. TensorList 算子

| 用例 | 验证特性 | 关键列 |
|------|---------|--------|
| `split_dyn_output` | 非原地 DYNAMIC_OUTPUT（6 张量 TensorList 输出）+ rank 未知 desc | `output_shapes`、`dyn_input_shapes` |
| `foreach_div_tensorlist` | 双 DYNAMIC_INPUT + 非原地 DYNAMIC_OUTPUT（逐元素 desc） | `input_shapes`、`dyn_input_shapes` |
| `foreach_add_inplace_tensorlist` | 原地 DYNAMIC_OUTPUT（输出与输入同名，框架自动识别 inplace） | `attributes`、`input_data_ranges` |
| `scatter_list_perf_only` | var 原地更新 + indice/updates/mask 混合 dtype；无 golden（perf-only） | `attributes`、`--golden-mode Disable` |
| `avgpool_update_int4` | int4 半字节打包输入 | `input_dtypes` |

> [!IMPORTANT]
> 除 `scatter_list_perf_only`（无 golden）外，其余用例的 golden 均由同目录 `golden.py` 插件提供（`--plugin` 加载）。
