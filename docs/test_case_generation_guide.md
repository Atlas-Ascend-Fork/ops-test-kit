# 用例生成

## 使用须知

本文介绍如何用表格文件定义 TTK 测试用例：文件结构、四种模式的字段定义，以及数据类型与字段值的写法。

> **说明**：测试模式由子命令决定，不会按表头自动切换；各模式执行命令见[任务执行](./task_execution_guide.md)。

文档导览如下：

1. **[文件结构](#一文件结构)**
2. **[公共字段](#二公共字段)**
3. **[专属字段](#三专属字段)**
4. **[类型与取值](#四类型与取值)**
5. **[参数化用例](#五参数化用例describe)**

## 一、文件结构

测试用例写在表格文件里：第一行为表头（列名），其余每行是一个用例。支持 CSV 与 Excel（.xlsx/.xlsm），两者字段定义等价，可互换；另有 `.zip` 容器可打包多个 csv 一次运行。

最小示例（Kernel 模式）：

```csv
testcase_name,op_name,input_shapes,input_dtypes,output_shapes,output_dtypes
add_01,add,"((128, 1024), (1, 1024))","('float32', 'float32')","((128, 1024),)","('float32',)"
```

### 1. CSV

- 逗号分隔的纯文本表格。
- 含括号、逗号、引号的字段必须用双引号包裹。

### 2. Excel（.xlsx/.xlsm）

- 单元格天然支持逗号、引号等字符，无需转义或加双引号。
- 默认读第一个工作表，`--sheet` 指定其他表：

  ```shell
  python3 -m ttk kernel -i cases.xlsx              # 首个工作表
  python3 -m ttk kernel -i cases.xlsx --sheet T2   # 指定工作表
  ```

### 3. 结果文件

无论输入是 csv 还是 xlsx，结果输出始终为 CSV：csv 输入命名 `{stem}_result.csv`，xlsx 输入命名 `{stem}_{sheet}_result.csv`，多 sheet 跑测互不覆盖。

## 二、公共字段

以下 13 个字段为四种模式共有（另可用 `attributes1`～`attributes9` 扩展属性列）。

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `testcase_name` | 自动生成 | 用例唯一名称（必填列）。整列缺失时自动生成 `auto_testcase_name_N`；某行留空则整表解析报错 |
| `input_data_ranges` | `((None, None),)` | 每输入的随机数据范围 `[min, max]`（闭区间，None 时 [-2, 2]）；min/max 及第 3 个及以上的值强制混入随机数据；min==max 生成固定值。支持 TensorList 嵌套 |
| `precision_tolerances` | `None` | 每输出的容差对 `(rtol, ptol)`，如 `"((0.001, 0.001),)"`。旧版字段（保留兼容），仅 `close`/`cosine`/`requant` 读取；默认 `mixed` 的容差在 TestSpec `tolerance` 中调整 |
| `absolute_precision` | `1e-8` | 默认绝对容差，可嵌套逐输出控制。旧版字段（保留兼容），仅 `close` 读取 |
| `is_enabled` | `True` | 设为 `False` 跳过此用例 |
| `remark` | `None` | 自由备注 |
| `soc_series` | `None` | SoC 过滤，前缀 `-` 表示排除。如 `('Ascend910A', '-Ascend310P')` |
| `priority` | `0` | 优先级，配合 `--priority` 筛选，非负整数 |
| `attributes` | `{}` | 算子/API 属性参数，语义随模式不同，详见各模式用例编写文档 |
| `device_ids` | `None` | 用例级设备指定（与 `--devices`/`--dev` 可用池取交集） |
| `batch_axis` / `batch_slice_info` / `batch_seed` | `None` | 批一致性三字段（all-or-nothing），详见[确定性计算与批一致性](./reference/determinism_compute.md) |

## 三、专属字段

四种模式在公共字段基础上另有专属字段（不含 `attributes1`~`attributes9` 扩展列）。

| 模式 | 子命令 | 核心标识字段 | 详见 |
|------|--------|------------|------|
| Kernel | `ttk kernel` | `op_name`、`input_shapes`、`output_shapes`、`attributes` | [Kernel用例编写](./operator_test_guides/kernel_case_writing_guide.md) |
| GEIR | `ttk geir` | `op_name`、`input_shapes`、`dyn_input_shapes` | [GEIR用例编写](./operator_test_guides/geir_case_writing_guide.md) |
| ACLNN | `ttk aclnn` | `api_name`、`tensor_view_shapes`、`output_tensor_indexes`、`scalar_dtypes` | [ACLNN用例编写](./operator_test_guides/aclnn_case_writing_guide.md) |
| E2E | `ttk e2e` | `api_name`、`tensor_view_shapes`、`golden_api` | [E2E用例编写](./operator_test_guides/e2e_case_writing_guide.md) |

> [!NOTE]
> GEIR 继承 Kernel 的字段，额外多 `dyn_input_shapes` 用于控制图编译时的动态 shape 描述。

> [!NOTE]
> E2E 内分 torch / tensorflow / onnx 三种 API，由 `api_name` 值区分（`onnx.<OpType>` 即 onnx），同一 CSV 不得混放不同框架 API。onnx 不支持布局列（`tensor_formats` / `tensor_storage_shapes` / `tensor_view_strides` / `tensor_view_offsets`）与动态 shape（维度 < 0），专属约定见 [E2E 用例编写](./operator_test_guides/e2e_case_writing_guide.md)。

## 四、类型与取值

### 1. 数据类型字符串

CSV 的 `input_dtypes`/`tensor_dtypes` 等列使用以下字符串，规范名与缩写等效：

| 数据类型 | 规范名 | 可用缩写 |
|---------|--------|---------|
| 16位浮点 | `float16` | `f16`, `fp16` |
| 32位浮点 | `float32` | `f32`, `fp32`, `float` |
| 64位浮点 | `float64` | `f64`, `fp64`, `double` |
| BFloat16 | `bfloat16` | `bf16`, `bfp16` |
| 8位有符号整数 | `int8` | `s8` |
| 16位有符号整数 | `int16` | `s16` |
| 32位有符号整数 | `int32` | `s32` |
| 64位有符号整数 | `int64` | `s64` |
| 8位无符号整数 | `uint8` | `u8` |
| 16位无符号整数 | `uint16` | `u16` |
| 32位无符号整数 | `uint32` | `u32` |
| 64位无符号整数 | `uint64` | `u64` |
| 布尔型 | `bool` | — |
| 32位复数 | `complex32` | `c32` |
| 64位复数 | `complex64` | `c64` |
| 128位复数 | `complex128` | `c128` |
| 低比特整数 | `int4` / `uint1` | `s4` / `u1`, `s1`, `int1` |
| 低比特浮点 | `float4_e2m1` | `f4_e2m1`, `fp4_e2m1` |
| 低比特浮点 | `float4_e1m2` | `f4_e1m2`, `fp4_e1m2` |
| 低比特浮点 | `float8_e8m0` | `f8_e8m0`, `fp8_e8m0` |
| 低比特浮点 | `float8_e5m2` | `f8_e5m2`, `fp8_e5m2` |
| 低比特浮点 | `float8_e4m3fn` | `f8_e4m3fn`, `fp8_e4m3fn` |
| 高精度浮点 | `hifloat8` | `hif8`, `hifp8` |
| 高精度浮点 | `hifloat4` | — |

### 2. 字段值写法

字段值按 Python 字面量书写；含括号、逗号、引号的值必须用双引号包裹：

| 字段值 | 写法 | 示例 |
|--------|------|------|
| 形状/格式元组 | 双引号包裹的嵌套元组 | `"((128, 1024), (1, 1024))"`、`"('ND', 'NCHW')"` |
| dtype 元组 | 双引号包裹的字符串元组 | `"('float32', 'float32')"` |
| TensorList 嵌套形状 | 三层嵌套元组 | `"(((3,3),(3,2)),(3,5))"` |
| 属性字典 | Python 字典 | `{'dim': -1}` |
| 数值/布尔 | 直接写 | `1e-8`、`0`、`True` |
| 空值 | 留空 | 两个连续逗号 `,,` |
| None 输入 | 写 `None` | `None` |

## 五、参数化用例（describe）

输入特别多的算子，可以不逐个声明形状，而在 CSV 里用自定义参数列（如 `B/S1/N1/D`）写关键参数，由算子 spec 的 `describe(params)` 把这些参数组装成 shapes/dtypes 回填用例：

```python
class GemmTestSpec:
    @staticmethod
    def describe(params):
        m, k, n = params["M"], params["K"], params["N"]
        return {"shapes": ((m, k), (k, n)), "dtypes": ("float16", "float16")}
```

对应 CSV：`testcase_name,api_name,M,K,N` → `gemm_tc,torch.mm,128,4096,4096`。

触发条件与完整契约见[TestSpec 规范](./reference/test_spec.md)。
