# 用例生成


---

# 简介

TTK使用表格文件（CSV 或 Excel .xlsx/.xlsm）批量定义测试用例。第一行为表头（列名），后续每行代表一个测试用例。不同测试模式使用不同的列定义。

**模式选择**（由子命令唯一决定，TTK 不按表头自动切换模式）：
- 表头包含 `api_name` 且值以 `aclnn` 开头 → 用 `ttk aclnn`（**ACLNN模式**）；`api_name` 不以 `aclnn` 开头（含 torch/tensorflow API 与 `onnx.<OpType>`）→ 用 `ttk e2e`（**E2E模式**，onnx 域见 [ONNX算子测试指南](./Operator_Test_Guides/ONNX_Test_Guide.md)）
- 表头不包含 `api_name` → 用 `ttk kernel`（**Kernel模式**）或 `ttk geir`（**GEIR模式**）

模式选错会导致用例解析失败。各模式的执行命令见 [任务执行](./Task_Execution.md)。

---

# 输入文件格式（CSV / Excel）

TTK 支持两种等价的输入文件格式，表头与字段定义完全相同（另有 `.zip` 容器可打包多个 csv，一次运行按多张表展开）：

| 格式 | 后缀 | 多工作表 | 说明 |
|------|------|---------|------|
| CSV | `.csv` | — | 文本表格，逗号分隔；含特殊字符（括号、逗号、引号）的字段须用双引号包裹。 |
| Excel | `.xlsx` / `.xlsm` | 支持 | 工作簿，每个工作表（Sheet）是一张表。 |

## Excel（.xlsx）用法

- **默认读取第一个工作表**；用 `--sheet` 指定工作表名，不存在时报错并列出可用工作表：
  ```shell
  python3 -m ttk kernel -i cases.xlsx              # 首个工作表
  python3 -m ttk kernel -i cases.xlsx --sheet T2   # 指定工作表
  ```
- **单元格一律按文本处理**：读取时每个单元格被转为字符串并去除首尾空白，与 CSV 单元格行为完全一致，后续字段解析逻辑（`input_shapes`、`attributes` 等的 `eval`）原样复用。因此 **xlsx 与 csv 的用例可互换**，字段回退、TensorList 嵌套等规则完全相同。
- **数值列建议设为文本格式**：Excel 会自动推断数字类型（如把 `01` 存为 `1`、`1e-8` 存为浮点）。为保证与 CSV 完全一致，建议把 `input_shapes`/`input_dtypes`/`attributes`/`input_data_ranges`/`output_shapes` 等列的单元格格式预设为「文本」。
- **空行处理**：整行空白的行会被跳过（与 CSV 空行一致）。
- **结果文件仍为 CSV**：无论输入是 csv 还是 xlsx，结果输出始终是 CSV，便于 diff 与版本管理。默认命名：csv 为 `{stem}_result.csv`；xlsx 为 `{stem}_{sheet}_result.csv`（`sheet` 为实际读取的工作表名，默认首页时也带上首页名，如 `cases_T1_result.csv`），同一文件多 sheet 跑测互不覆盖。

> CSV 中「含特殊字符的字段须双引号包裹」的规则不适用于 xlsx——Excel 单元格天然支持逗号、引号等字符，无需转义。

---

# 公共字段（所有模式通用）

以下13个字段为Kernel、GEIR、ACLNN、E2E四种模式共有。

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `testcase_name` | STRING | **是** | 自动生成 | 用例唯一名称。仅当整列缺失时自动生成为 `auto_testcase_name_N`；列存在但某行留空会导致整表解析报错。 |
| `input_data_ranges` | SHAPELIKE_FLOAT_SIGNED_NESTED | 否 | `((None, None),)` | 每个输入张量的随机数据范围。每个元素为 `[min, max]`（闭区间，min 和 max 为边界值），None 时 min=-2、max=2。min/max 边界值与第 3 个及以上的值均为必现值，会强制混入随机数据中保证出现（所有必现值合计约占 tensor 规模的 25%，即每个值约 25%÷必现值个数；大 tensor 每值封顶 2048 次，避免边界值淹没正常分布）。min==max 时生成固定值。支持TensorList嵌套。如 `"((-1, 1), (0, 10))"` 或 `"((-1, 1, 0), (0, 10))"` |
| `precision_tolerances` | SHAPELIKE_FLOAT_NESTED | 否 | `None` | 每个输出的精度容差对 `(rtol, ptol)`。如 `"((0.001, 0.001),)"`。legacy 字段，仅 `--compare close`/`cosine`/`requant` 读取（requant 只读其中的 ptol，作为允许超差元素占比；经 TestSpec `tolerance` 声明的 `quant` 判据同样读取 ptol）；默认 `mix_tolerance` 的容差在 TestSpec `tolerance` 中调整 |
| `absolute_precision` | FLOAT_SCALAR_NESTED | 否 | `1e-8` | 默认绝对精度容差。可以是单个浮点数或嵌套容器实现逐输出控制。legacy 字段，仅 `--compare close` 读取 |
| `is_enabled` | BOOL | 否 | `True` | 设为 `False` 跳过此用例。 |
| `remark` | STRING | 否 | `None` | 自由备注信息。 |
| `soc_series` | STRING_CONTAINER | 否 | `None` | SoC过滤。前缀 `-` 表示排除。如 `('Ascend910A', '-Ascend310P')` |
| `priority` | INT | 否 | `0` | 优先级，用于选择性执行。非负整数。 |
| `attributes` | DICT | 否 | `{}` | 算子/API 属性参数，语义随模式不同（Kernel 为算子属性，ACLNN 为标量参数，E2E 为 API 参数），详见各模式用例编写文档。 |
| `device_ids` | STRING | 否 | `None` | 用例级设备指定（与 `--devices`/`--dev` 可用池取交集）。 |
| `batch_axis` / `batch_slice_info` / `batch_seed` | — | 否 | `None` | 批一致性三字段（all-or-nothing），详见[确定性计算与批一致性](./Deterministic_Compute.md)。 |

---

# 各模式特殊字段

四种模式在公共字段基础上各有专属字段（不含 `attributes1`~`attributes9` 扩展列）。

| 模式 | 子命令 | 专属字段数 | 核心标识字段 | 详见 |
|------|--------|-----------|------------|------|
| Kernel | `ttk kernel` | 17 | `op_name`、`input_shapes`、`output_shapes`、`attributes` | [Kernel用例编写](./Operator_Test_Guides/Kernel_Case_Writing.md) |
| GEIR | `ttk geir` | 18 | `op_name`、`input_shapes`、`dyn_input_shapes` | [GEIR用例编写](./Operator_Test_Guides/GEIR_Case_Writing.md) |
| ACLNN | `ttk aclnn` | 15 | `api_name`、`tensor_view_shapes`、`output_tensor_indexes`、`scalar_dtypes` | [ACLNN用例编写](./Operator_Test_Guides/ACLNN_Case_Writing.md) |
| E2E | `ttk e2e` | 10 | `api_name`、`tensor_view_shapes`、`golden_api` | [E2E用例编写](./Operator_Test_Guides/E2E_Case_Writing.md) |

> GEIR 继承 Kernel 的字段，额外多 `dyn_input_shapes` 用于控制图编译时的动态 shape 描述。

> E2E 内分 torch/tensorflow/onnx 三个域，由 `api_name` 值区分（`onnx.<OpType>` 即 onnx 域），**同一 CSV 不得混域**（worker 进程预载互斥：torch 进程预载 torch 系模块、tf 进程预载 tensorflow、onnx 进程预载 numpy + onnx）。onnx 域不支持布局列（`tensor_formats` / `tensor_storage_shapes` / `tensor_view_strides` / `tensor_view_offsets`，填写报 `ONNX_LAYOUT_UNSUPPORTED`）与动态 shape（dim < 0，即 -1 槽位，校验期报 `ONNX_DYNAMIC_INPUT_UNSUPPORTED`；仅支持静态 shape，dim ≥ 0），专属约定详见 [ONNX用例编写约定](./Operator_Test_Guides/ONNX_Test_Guide.md#测试用例编写)。

---

# 数据类型字符串

常用数据类型如下（另支持 `int4`/`uint1`/`float4_e2m1`/`float4_e1m2`/`float8_e8m0`/`float8_e5m2`/`float8_e4m3fn`/`hifloat8`/`hifloat4` 等，完整映射见 `ttk/utilities/dtypes.py` 的 `DTYPE_MAP`）：

| 数据类型 | 字符串表示 |
|---------|-----------|
| 16位浮点 | `float16`, `fp16` |
| 32位浮点 | `float32`, `fp32` |
| 64位浮点 | `float64`, `fp64` |
| BFloat16 | `bfloat16`, `bf16` |
| 8位有符号整数 | `int8` |
| 16位有符号整数 | `int16` |
| 32位有符号整数 | `int32` |
| 64位有符号整数 | `int64` |
| 8位无符号整数 | `uint8` |
| 16位无符号整数 | `uint16` |
| 32位无符号整数 | `uint32` |
| 64位无符号整数 | `uint64` |
| 布尔型 | `bool` |
| 32位复数 | `complex32`, `c32` |
| 64位复数 | `complex64`, `c64` |
| 128位复数 | `complex128`, `c128` |

# CSV字段值格式参考

| 类型 | 格式 | 示例 |
|------|------|------|
| shape元组 | 双引号包裹的嵌套元组 | `"((128, 1024), (1, 1024))"` |
| dtype元组 | 双引号包裹的字符串元组 | `"('float32', 'float32')"` |
| 嵌套shape（TensorList） | 双引号包裹的三层嵌套元组 | `"(((3,3),(3,2)),(3,5))"` |
| 字典 | 直接写Python字典 | `{'dim': -1}` |
| 数值 | 直接写数值 | `1e-8`, `0`, `True` |
| 空值 | 留空 | 两个连续逗号 `,,` |
| None输入 | `None` 关键字 | `None` |

> **注意**：包含特殊字符（括号、逗号、引号）的字段必须用双引号包裹。

# 参数化用例（describe）

输入特别多的算子（如 18 输入的 sparse_flash_mla）可以用抽象参数列替代逐张量
形状声明：在 CSV 里直接写 `B/S1/N1/D` 等待识别列，在算子 spec 里提供
`describe(params)`，由它组装 shapes/dtypes 回填用例。

- 触发条件：该行 CSV 的**形状/类型字段任一缺省**（kernel/geir 为
  `input_shapes`/`input_dtypes`，ACLNN/E2E 为 `tensor_view_shapes`/
  `tensor_dtypes`）。两者都已显式声明时，未识别列照旧忽略，行为不变。
- `params`：该行所有 TTK 未识别列的非空值汇总。数字/容器经字面量求值
  （`2`→int、`(128, 64)`→tuple），裸词（`BSND`、`bfloat16`）退回字符串。
- **契约**：`describe` 只返回 `{"shapes": ..., "dtypes": ...}` 两个 key，
  框架按通路映射到本通路 shape 字段（kernel/geir → `input_shapes`/
  `input_dtypes`，ACLNN/E2E → `tensor_view_shapes`/`tensor_dtypes`）。
  ACLNN 的 shapes 按 API 参数列表**含输出槽位**。
- **缺啥补啥**：shapes 与 dtypes 不强绑定——任一缺省即触发 describe，
  只回填缺省字段；CSV 已显式声明的字段不回填（"不缺的不读"）。
- 其余字段不参与契约：`attributes`、`output_tensor_indexes`、
  `output_shapes`/`output_dtypes` 等仍由 CSV 列显式提供（可写 `ELEWISE`
  等推断字符串）；CSV 显式声明的字段优先，describe 不覆盖。
- 失败语义：describe 抛异常/非 dict/缺 `shapes` → 用例置无效
  （`DESCRIBE_FAILED` / `DESCRIBE_RESULT_NOT_DICT` / `DESCRIBE_MISSING_SHAPES`）。

```python
class GemmTestSpec:
    @staticmethod
    def describe(params):
        m, k, n = params["M"], params["K"], params["N"]
        return {
            "shapes": ((m, k), (k, n)),
            "dtypes": ("float16", "float16"),
        }
```

对应 CSV：`testcase_name,api_name,M,K,N` → `gemm_tc,torch.mm,128,4096,4096`。

# 回退链

部分字段在未显式设置时自动回退到其他字段：| 字段 | 回退到 |
|------|--------|
| `input_ori_shapes` | `input_shapes` |
| `output_ori_shapes` | `output_shapes` |
| `input_ori_formats` | `input_formats` |
| `output_ori_formats` | `output_formats` |
| `input_formats` | *(无回退，默认 `('ND',)`)* |
| `output_formats` | `output_ori_formats` |
