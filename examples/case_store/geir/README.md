# GEIR 通路示例用例

单文件 `cases.csv` 覆盖 GEIR 通路的主要功能面，每条用例的 `remark` 列即测试项；`golden.py` 为配套 golden 插件（`--plugin` 加载，与 Kernel 共用同一套 TestSpec 契约）。GEIR 的 CSV 字段与 Kernel 完全一致，额外多一列 GEIR 专属的 `dyn_input_shapes`（图编译 desc_shape，仅在 `-d` 动态编译模式下生效；未指定时等价于将 `input_shapes` 全部维度替换为 `-1`）。

```bash
G=examples/case_store/geir
# 全量 golden 用例（12 例，静态 + 动态两种图编译模式一次跑齐）
python3 -m ttk geir -i $G/cases.csv --plugin $G/golden.py -c -d --no-op scatter_list

# perf-only 用例（无 golden，执行/性能验证；int8 输出走 binary_equal 路由，Disable 生效）
python3 -m ttk geir -i $G/cases.csv -c -d --golden-mode Disable -t scatter_list_perf_only
```

> GEIR 默认开启静态编译（`-c`），与 Kernel 默认开启动态编译相反；`dyn_input_shapes` 仅在显式 `-d` 时参与图构建。
>
> **已知限制**：`concat_d`（DYNAMIC_INPUT 聚合）在 `-d` 动态编译下图执行期 `EZ1001` shape 校验失败（GE 侧），当前仅 `-c`——该用例经 Kernel 用例矩阵以 `-c` 复用覆盖；其余 TensorList 算子（split / foreach / scatter）全模式可用。

## dyn_input_shapes 语义

| 用例 | 测试项 |
|---|---|
| `add_dyn_default` | 列缺省：动态编译时全维度自动替换为 `-1`（与显式全 `-1` 等价的对照例） |
| `add_dyn_tail_fixed` | 尾维固定 `((-1, 7188), (-1,))`：拼接维动态、特征维编译期固定 |
| `add_dyn_mixed_bcast` | 混合固定/动态维度 + 广播输入整体固定 `((128, -1), (1, 1024))` |
| `add_dyn_rank_unknown` | **rank 未知**：desc_shape `(-2,)` 表示未知秩（算子须 dynamicRankSupport），广播 infershape 照常 |

```bash
python3 -m ttk geir -i $G/cases.csv --plugin $G/golden.py -d \
    -t add_dyn_default,add_dyn_tail_fixed,add_dyn_mixed_bcast,add_dyn_rank_unknown
```

## 可选输入与复杂属性

| 用例 | 测试项 |
|---|---|
| `conv2d_nchw_optional` | NCHW 格式 + 复杂属性（strides/pads/dilations/groups/data_format）+ bias/offset_w 可选输入 None + 仅数据 N/H/W 动态、filter 固定 |
| `arg_max_v2_range_pin` | ValueDepend 张量输入（dimension）经 `input_data_ranges` 定值（`(1,1)` 生成常量 1），不经 attributes 指定 |
| `arg_max_v2_dtype_attr` | **Type 型属性**（proto `ATTR(dtype, Type)`）字符串值 `'int32'` 经 StringToDataType 按枚举 int 落位 int64 字段（曾因工厂 setter 落 DataType 字段致 tbe 侧 `E29999` 取值失败） |

```bash
python3 -m ttk geir -i $G/cases.csv --plugin $G/golden.py -c -d \
    -t conv2d_nchw_optional,arg_max_v2_range_pin,arg_max_v2_dtype_attr
```

## TensorList 算子（split / foreach / scatter / avgpool）

| 用例 | 测试项 |
|---|---|
| `split_dyn_output` | 常用算子**非原地** DYNAMIC_OUTPUT（6 张量 TensorList 输出）+ rank 未知 desc + split_dim ValueDepend 经 attributes 同名提取——历史缺陷回归：逐元素输出 dtype 置 `DT_UNDEFINED` 曾致 `E39999`（现落 CSV 声明 dtype） |
| `foreach_div_tensorlist` | 双 DYNAMIC_INPUT + 非原地 DYNAMIC_OUTPUT（逐元素 desc）+ 嵌套 `dyn_input_shapes` 逐元素 rank 未知 `((-2,)×N)` |
| `foreach_add_inplace_tensorlist` | **原地** DYNAMIC_OUTPUT（输出名与输入 x1 同名自动识别 `output_inplace_indexes`，输出 desc 逐元素继承输入——`E40021` 历史缺陷回归）+ 标量 alpha 张量输入经值域定值 |
| `scatter_list_perf_only` | TensorList var 原地更新 + indice(int64)/updates(int8)/mask(uint8 可选) 混合 dtype + 嵌套 rank 未知 desc；无 golden（算子语义未公开，perf-only） |
| `avgpool_update_int4` | **int4 输入**（半字节 2 元素/字节打包，`(n+1)/2` 字节对齐——历史缺陷回归：缺 `DT_INT4` 映射时静默回退 `DT_FLOAT`，输入文件 4 倍大小校验失败）；y = x1/(kh·kw)（pads=0 时 x2 为 padding 掩码通道不影响输出，golden 经探针实证） |

```bash
python3 -m ttk geir -i $G/cases.csv --plugin $G/golden.py -c -d \
    -t split_dyn_output,foreach_div_tensorlist,foreach_add_inplace_tensorlist,avgpool_update_int4
python3 -m ttk geir -i $G/cases.csv -c -d --golden-mode Disable -t scatter_list_perf_only
```

## 历史缺陷回归

| 用例 | 测试项 |
|---|---|
| `smoothl1_loss_inf_attr` | **非有限浮点属性**（`{'sigma': inf}`）：config JSON 序列化为 `Infinity` 非严格 token，C++ 侧需解析为 `HUGE_VAL`（曾因不识别该 token 致其后键值对整体丢失）；sigma=inf 时恒走二次分支 y≡0 |
| `arg_max_v2_dtype_attr` | Type 型属性 int64 落位（见上表） |
| `split_dyn_output` | DYNAMIC_OUTPUT 声明 dtype 落位（见上表） |
| `foreach_add_inplace_tensorlist` | inplace TensorList 输出 desc 继承（见上表） |
| `avgpool_update_int4` | int4 dtype 映射与半字节打包 size 校验（见上表） |

```bash
python3 -m ttk geir -i $G/cases.csv --plugin $G/golden.py -c -d \
    -t smoothl1_loss_inf_attr,arg_max_v2_dtype_attr,split_dyn_output
```

## 复用 Kernel 用例

Kernel 的用例矩阵可直接跑 GEIR（字段超集兼容，无需改动；`concat_d` 仅 `-c`，见上方已知限制）：

```bash
K=examples/case_store/kernel
python3 -m ttk geir -i $K/cases.csv --plugin $K/golden.py -c -d -t add_fp16_broadcast,abs_int16_3d,add_complex64,pad_float8,pad_float4
# concat_d 仅 -c（见上方已知限制）
python3 -m ttk geir -i $K/cases.csv --plugin $K/golden.py -c -t concat_d_tensorlist_in
```

dtype 覆盖（geir 原生 + Kernel 矩阵复用）：float16 / float32 / bfloat16 / int8 / int16 / int32 / int64 / **int4** / **uint8** / complex64 / float8_e5m2 / float4_e2m1。

用例编写与命令行细节见 `.claude/skills/ttk-how-write-case/` 与 `ttk-how-run-test`；`dyn_input_shapes` 字段说明见 [GEIR 用例编写](../../../docs/Operator_Test_Guides/GEIR_Case_Writing.md)。
