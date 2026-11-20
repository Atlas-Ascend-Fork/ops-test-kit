# GEIR 图融合Pass

## 使用须知

`graph_ops` 列非空时，该用例即为图模式用例（多算子 DAG 建图）：`ttk geir` 依据 `graph_ops`、`graph_edges`、`graph_outputs` 三列，将用例组装为多算子 GE 计算图并整体编译执行，使融合 pass 的源模式在图编译期真实出现，支持以下两类验证：

- 融合触发验证：通过结果列 `fusion_pass_details` 获取各 pass 的命中（hit）与生效（effect）计数，取值按编译模式组装（`cst:xxx,dyn:xxx`，两模式触发可不同）。
- 数值等价验证：整图输出与 golden 计算结果进行精度比对。

`graph_ops` 列为空时，图模式行为不生效，同一 CSV 文件中可以混放单算子用例与图模式用例。

图融合Pass 的执行命令见 [GEIR 算子测试指南](../operator_test_guides/geir_test_guide.md#图融合pass)；本文档为图模式用例的详细参考，用例各列的基础用法见 [GEIR 用例编写](../operator_test_guides/geir_case_writing_guide.md)。

> [!NOTE]
> 使用图模式前，须确认以下条件：
>
> - 图中各节点的算子已部署且可见（自定义算子须先安装对应的 .run 包并 source 环境变量）。
> - 待验证的融合 pass 已部署在 GE 可发现路径下。pass 未部署时图可正常执行，但融合计数列为 NA。

## 文档导览

1. **[功能概述](#一功能概述)**
2. **[专属字段](#二专属字段)**
3. **[引用格式](#三引用格式)**
4. **[校验规则](#四校验规则)**
5. **[属性格式](#五属性格式)**
6. **[插件编写](#六插件编写)**

## 一、功能概述

图模式用例通过 `graph_ops` 列声明。该列填写构成 DAG 的节点列表（如 `"('mul','add_n')"`），列缺省或为空时，用例按单算子模式执行。节点列表中的元素为算子名，须按拓扑序排列，即数据从低序节点流向高序节点；同一算子允许重复出现。

图模式下，`op_name` 列仅用于关联 golden 插件与容差，可填写融合目标算子名或图中的代表算子名；算子平台过滤由 `soc_series` 列控制。

图模式校验不依赖设备环境，可通过 `python3 -m ttk geir -i <csv> --validate` 单独校验用例。校验未通过的用例将被标记为 `GRAPH_DESC_INVALID`，此类用例不执行。

## 二、专属字段

| 列 | 说明 | 缺省行为 |
|----|------|----------|
| `graph_ops` | 节点列表，须按拓扑序声明 | 留空时不启用图模式 |
| `graph_edges` | `(src, dst)` 形式的边元组序列 | 留空仅限单节点图，见第三节 |
| `graph_outputs` | 图输出引用，格式同边引用（`'oN:port[:K]'`，不允许引用 `dN`） | 留空时取末节点的全部输出；展开项数须与 `output_shapes`、`output_dtypes` 一致 |

## 三、引用格式

边引用的格式如下：

```text
dN          图输入槽，唯一输出，不带端口名
oN:port     节点 N 的固定输出/输入口，端口名取自该算子定义，大小写敏感
oN:port:K   动态口（DYNAMIC_INPUT/DYNAMIC_OUTPUT）的第 K 个元素，K 从 0 起计数
```

填写规则：

- `dN` 中的 `N` 为该输入在 `input_shapes` 中的位序；`oN` 中的 `N` 为该节点在 `graph_ops` 中的位序；`d`、`o` 为引用保留前缀。
- 源侧只能为 `dN` 或算子输出口；目的侧只能为算子输入口。
- 固定口不允许带 `:K`，动态口必须带 `:K`；端口名必须存在于该算子对应方向的定义中。
- 同一个目的槽（节点、口、元素）至多一条入边；源扇出合法，即同一个源可以连接多条边（如 `d0` 同时连接 `o0:x1` 与 `o0:x2`）。
- `graph_edges` 留空仅限单节点图：各必选口按声明序依次接入 `d0, d1, …`；多节点图须显式声明全部连线。

## 四、校验规则

图模式校验在下发执行前完成，未通过的用例将标记 `GRAPH_DESC_INVALID`，不执行；错误信息为英文描述。校验内容如下：

- 必选输入口恰有一条入边；可选口至多一条；同一目的槽不重复入边
- 引用格式、端口名、序号均合法：源侧为 `dN` 或算子输出口，目的侧为算子输入口；固定口禁 `:K`、动态口必须 `:K`
- 节点须按拓扑序声明，边只允许低序到高序
- 每个图输入至少被一条边引用
- 定值口有边且源为 `dN`，同一个 `dN` 的多处定值一致
- 动态输出口全部被引用（边或 `graph_outputs`），元素序号从 0 连续

其余错误前缀：`proto:`（节点算子不可见）、`edges:`（graph_edges 留空的默认接线不适用，须显式声明全部连线）、`attrs:`（`attributes` 嵌套格式填写错误）、`outputs:`（图输出引用非法或数量不符）、`flat:`（图输入声明非法）、`mode:`（命中使用限制，如推理字符串、inplace、batch 列、`attributesN`）。

## 五、属性格式

图模式下，`attributes` 列使用嵌套格式声明：顶层键用于指定节点，值用于声明该节点的属性与常量。

```csv
"{'o0': {'axes': (1,), 'dtype': 'float32'}, 'o1': {'N': 2}}"
```

- 顶层键为节点引用 `oN`，也可以使用该节点的算子名；同一算子出现多次时不可使用算子名，须使用 `oN`。值必须为字典。
- 值内的键分为以下几类：

| 键形态 | 含义 |
|--------|------|
| 算子属性名（ATTR / REQUIRED_ATTR） | 该节点的属性，其中 REQUIRED_ATTR 声明的属性必须给出 |
| 输入口名（值内，同名判断） | 常量定值：该输入以 Const 节点进入计算图，其源必须是 `dN`；该输入的 `input_data_ranges` 不生效 |
| `!` / `#` / `@` 前缀键 | 内部属性，用法与单算子模式一致 |
| 其余（含空字符串与非字符串键） | 非法键，校验报错 |

- `attributes1`～`attributes9` 填写后校验报错，节点属性须全部通过嵌套 `attributes` 声明。

### Type 类属性的两种写法

类型为 Type/DataType 的属性（如 `dtype`、`dst_type`）有两种合法写法，两种写法在同一部署环境下不可互换，须根据算子实现对属性的读取方式选择：

| 写法 | 示例 | 适用算子 |
|------|------|----------|
| 字符串 | `'int32'` / `'DT_INT32'` | infershape 或融合 pass 从属性中直接读取 dtype 的算子（如 `arg_max_with_value` 的 `indice_dtype`） |
| 整数（GE 枚举值） | `0`（DT_FLOAT）、`3`（DT_INT32） | 按整型读取属性的历史算子（如 `cast` 的 `dst_type`） |

- 写法与算子的读取方式不匹配时，任务将以 E29999（图编译失败）或 E89999（执行失败）报错，不会产生错误的计算结果。
- 字符串值不在下表范围内时，将按 `DT_FLOAT` 处理且不报错，因此字符串值须使用下表中的名称：

| 数据类型 | GE 枚举 | 数据类型 | GE 枚举 |
|--------|---------|--------|---------|
| `float32` | `DT_FLOAT` | `uint1` | `DT_UINT1` |
| `float16` | `DT_FLOAT16` | `float8_e4m3fn` | `DT_FLOAT8_E4M3FN` |
| `bfloat16` | `DT_BF16` | `float8_e5m2` | `DT_FLOAT8_E5M2` |
| `int64` | `DT_INT64` | `float8_e8m0` | `DT_FLOAT8_E8M0` |
| `int32` | `DT_INT32` | `hifloat8` | `DT_HIFLOAT8` |
| `int16` | `DT_INT16` | `float4_e2m1` | `DT_FLOAT4_E2M1` |
| `int8` | `DT_INT8` | `float4_e1m2` | `DT_FLOAT4_E1M2` |
| `int4` | `DT_INT4` | `hifloat4` | `DT_HIFLOAT4` |
| `uint64` | `DT_UINT64` | `hifloat4_scale` | `DT_HIFLOAT4_SCALE` |
| `uint32` | `DT_UINT32` | `complex32` | `DT_COMPLEX32` |
| `uint16` | `DT_UINT16` | `complex64` | `DT_COMPLEX64` |
| `uint8` | `DT_UINT8` | `complex128` | `DT_COMPLEX128` |
| `float64` / `double` | `DT_DOUBLE` | `bool` | `DT_BOOL` |

## 六、插件编写

图模式与单算子使用同一套 TestSpec 插件契约，通过 `--plugin` 加载，注册名为用例的 `op_name`，契约详见 [TestSpec — 算子测试规范](./test_spec.md)。图模式下的规则：

- `*args` 为按图输入槽序（`dN`）传入的输入，不含 `None` 与嵌套结构；定值槽对应的输入已被定值覆盖，可直接作为数值使用。
- `kwargs["output_dtypes"]` 等元信息为图级口径；`attributes` 嵌套格式按顶层键出现在 kwargs 中，值为对应节点的字典。
- golden 按该用例声明的连线，使用 numpy/torch 实现整图的等价计算，将各 `dN` 代入公式。
- 同一个 `op_name` 不得对应不同的连线；不同连线须使用不同的 `op_name` 分别注册。
- 容差与 `op_name` 关联（`tolerance` 列 / Spec）。

参考实现见 `examples/case_store/geir/golden.py`。
