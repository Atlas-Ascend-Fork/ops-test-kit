# Kernel 模式用例编写

子命令：`python3 -m ttk kernel`
结构类：`TestcaseKernel`（标准字段 30 个：13 公共 + 17 专有；另有 `attributes1`～`attributes9` 扩展列，全量 39 个）

## 用例配置依据

输入/输出的**数量和顺序**由算子定义文件决定，必须严格一一对应。属性的**名称和可用项**同样来自这些文件，但属性放在 dict 中，无顺序要求。查看优先级：

1. **算子定义**：`xxx_def.cpp`（如 `add_def.cpp`），定义 INPUT/OUTPUT/ATTR 声明
2. **算子信息库**：`aic-{芯片系列号}-ops-info-{算子仓[4:]}.json`，包含每个输入/输出的 name、支持 dtype、支持 format、paramType（optional/dynamic）、ValueDepend 等
  - builtin 算子信息库路径：`{ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/config/{芯片系列号}/aic-{芯片系列号}-ops-info-{算子仓[4:]}.json`
  - 自定义算子信息库路径：`{ASCEND_OPP_PATH}/vendors/customize/op_impl/ai_core/tbe/config/{芯片系列号}/aic-{芯片系列号}-ops-info.json`
  - 例如 add 算子在 `ascend910b` 芯片上的 builtin 信息库：`$ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/config/ascend910b/aic-ascend910b-ops-info-math.json`

## 必填字段

| 字段 | 说明 |
|------|------|
| `op_name` | 算子名，优先使用算子信息库中 `opInterface.value` 的值（下划线格式）。若未配置 opInterface，则将驼峰算子名转为下划线格式即可。如 `add`、`mat_mul_v3`、`reduce_min` |
| `input_shapes` | 输入 shape，按算子定义的输入顺序排列。如 `"((128,1024),(1,1024))"` |
| `input_dtypes` | 输入 dtype，与 input_shapes 一一对应。如 `"('float32','float32')"`。支持广播：`"('float32',)"` 填充所有位置 |
| `output_shapes` | 输出 shape，按算子定义的输出顺序排列。如 `"((128,1024),)"` |
| `output_dtypes` | 输出 dtype，与 output_shapes 一一对应。支持广播：`"('float32',)"` 填充所有位置 |

## 完整字段表

### 格式与原始 Shape

| 字段 | 必填 | 默认 | 回退到 | 说明 |
|------|------|------|--------|------|
| `input_formats` | 否 | `('ND',)` | *(无)* | 输入格式。如 `('ND','NCHW')`。支持广播：`"('ND',)"` 填充所有位置 |
| `input_ori_shapes` | 否 | → input_shapes | input_shapes | 格式转换前的原始 shape |
| `input_ori_formats` | 否 | `('ND',)` | input_formats | 原始输入格式。支持广播 |
| `output_formats` | 否 | `('ND',)` | output_ori_formats | 输出格式。支持广播 |
| `output_ori_shapes` | 否 | None | output_shapes | 原始输出 shape |
| `output_ori_formats` | 否 | `('ND',)` | output_formats | 原始输出格式。支持广播 |

### 属性与特殊字段

| 字段 | 必填 | 默认 | 说明 |
|------|------|------|------|
| `attributes` | 否 | `{}` | 算子属性。如 `{'transpose_x1': True, 'transpose_x2': False}` |
| `output_inplace_indexes` | 否 | `()` | 原地操作输出索引。无需手动填写：框架自动从算子信息库推断（当输出名称与输入名称相同时，表示输出覆写到该输入内存上） |
| `output_shape_unknown_indexes` | 否 | `()` | 编译期 shape 未知的**输出张量索引**。如 `"(0,)"` 表示第 0 个输出的 shape 在编译期无法确定 |

### Dump 与 指定数据

| 字段 | 必填 | 默认 | 说明 |
|------|------|------|------|
| `dump_file_prefix` | 否 | None | Dump 文件名前缀 |
| `manual_input_binaries` | 否 | `()` | 手动输入数据文件路径。支持 `.npy`、`.pt`、裸二进制格式。Python 表达式 |
| `manual_golden_binaries` | 否 | `()` | 手动 Golden 数据文件路径。支持 `.npy`、`.pt`、裸二进制格式。Python 表达式 |
| `manual_xpu_binaries` | 否 | `()` | 手动 XPU 输出文件路径。`cross_check` 时优先于远端 XPU，支持 `.npy`、`.pt`、裸二进制格式。配置后必须同时提供手动输入和 Golden。Python 表达式 |

### 回退链

| 字段 | 回退到 |
|------|--------|
| `input_ori_shapes` | `input_shapes` |
| `output_ori_shapes` | `output_shapes` |
| `input_ori_formats` | `input_formats` |
| `output_ori_formats` | `output_formats` |
| `output_formats` | `output_ori_formats` |

## 常见模式

### TensorList 嵌套

何时使用 TensorList 格式：当算子信息库中 ParamType = DYNAMIC 时，即使只包含 1 个张量也必须用 TensorList 嵌套格式。

三层嵌套表示 TensorList 分组：外层括号分组，内层括号是单个张量的 shape。

```csv
input_shapes,"(((3,3),(3,2)),(3,5))"
```

含义：TensorList(2 个张量 3x3 和 3x2) + 1 个普通张量 3x5。

### 动态 Shape

框架自动将正数维度替换为 `-1`，用户在 `input_shapes` 写具体 shape 即可。动态编译开关：`-d`。

### 可选输入

当算子信息库中某个输入的 paramType 为 optional 时，该输入是可选输入，若当前用例中该 Tensor 不传递，则对应的 shape/dtype/format 都用 `None` 占位：

```csv
input_shapes,"((128,1024),None,None)"
input_dtypes,"('float32',None,None)"
input_formats,"('ND',None,None)"
```

### Inplace 算子

框架自动从算子信息库推断（当输出名称与输入名称相同时，表示输出覆写到该输入内存）。仅当无法自动推断时才需手动指定，值为被覆写的**输入张量索引**：

```csv
output_inplace_indexes,"(0,)"    # 第 0 个输入被覆写
output_inplace_indexes,"(0,2)"   # 第 0 和第 2 个输入被覆写
```

### 通过 attributes 指定张量值

在 `attributes` 字典中按算子定义的参数名指定张量值，适用于以下场景：

**场景 1：ValueDepend 张量输入（必须指定）**

当算子信息库中某个输入的 ValueDepend 为 OPTIONAL 或 REQUIRED 时，**必须**在 attributes 中指定该张量的值。张量本身仍作为普通输入出现在 input_shapes/input_dtypes 中。

例如 ReduceMin 的 `axes` (ValueDepend=optional)

**场景 2：非 ValueDepend 张量的值约束（可选）**

某些输入虽然不是 ValueDepend 类型，但其数值不能由框架随机生成，常见于索引参数、shape 大小参数、size 参数等——这些值与其他输入的 shape 有关联。有两种选择：a) 写输入定制函数；b) 在 attributes 中指定张量值。对于小 shape 张量，b) 方案更便捷。

例如 grouped_matmul 的 `group_list` 表示分组大小，其值与输入 shape 关联，不能随机：

```csv
attributes,"{'group_list': [1, 1, 1]}"
```

此外，也可以单纯为了测试目的指定某个张量的具体值，而不用 `manual_input_binaries` 传入整个输入文件。

## 示例

### 基本示例

```csv
testcase_name,op_name,input_shapes,input_dtypes,output_shapes,output_dtypes,attributes
add_01,add,"((128,1024),(128,1024))","('float32','float32')","(128,1024)","('float32',)",
```

### 带属性和可选输入（MatMulV3）

MatMulV3 有 4 个输入，其中第 3、4 个为可选输入（bias 和 offset），不传递时用 `None` 占位：

```csv
testcase_name,op_name,input_shapes,input_dtypes,output_shapes,output_dtypes,attributes
matmul_01,mat_mul_v3,"((512,1792),(1792,256),None,None)","('bfloat16','bfloat16',None,None)","(512,256)","('bfloat16',)","{'transpose_x1':False,'transpose_x2':False,'offset_x':0}"
```

### ValueDepend 张量输入（ReduceMin）

ReduceMin 的 `axes` 是 ValueDepend=Optional 的张量输入，**必须**在 attributes 中指定其值：

```csv
testcase_name,op_name,input_shapes,input_dtypes,output_shapes,output_dtypes,attributes
reduce_min_00,reduce_min,"((512,),(1,))","('float32', 'int32')","((1,),)","('float32',)","{'axes': [0], 'keepdims': False}"
```

更多示例见 `examples/case_store/kernel/`。
