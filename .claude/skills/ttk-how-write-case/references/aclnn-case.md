# ACLNN 模式用例编写

子命令：`python3 -m ttk aclnn`
结构类：`TestcaseAclnn`（标准字段 28 个：13 公共 + 15 专有；另有 `attributes1`～`attributes9` 扩展列，全量 37 个）

## 用例配置依据

ACLNN 用例的参数（张量数量/顺序/dtype、标量、属性等）来自 ACLNN 接口定义头文件。查看优先级：
1. **开发仓源码**：`aclnn_xx.h`
2. **安装路径头文件**：`{ASCEND_HOME_PATH}/include/aclnnop/aclnn_xx.h`

`aclnnXXGetWorkspaceSize` 接口定义包含完整的参数列表和类型信息

## 配置原则

`aclnnXXGetWorkspaceSize` 中的参数按类型归类到对应 CSV 字段：

| 参数类型 | 归类字段 | 说明 |
|----------|----------|------|
| `aclTensor*` / `aclTensorList*` | `tensor_view_shapes`, `tensor_dtypes`, `tensor_formats` 等 | 张量参数，按接口声明顺序排列。TensorList 用嵌套格式 |
| `aclScalar*` / `aclScalarList*` | `scalar_dtypes` + `attributes` | 标量参数，dtype 填 scalar_dtypes，值默认随机生成（由 `scalar_data_ranges` 控制范围），也可在 `attributes` 中指定具体值（非必须） |
| `int64_t`, `float`, `double`, `bool`, `char*` 等 C 基本类型 | `attributes` | 直接作为属性值 |
| `aclnnIntArray*` 等 `aclnn*Array` | `attributes` | 列表型属性，值填 list |

## 必填字段

| 字段 | 说明 |
|------|------|
| `api_name` | ACLNN API 名。取头文件接口名 `aclnnXXGetWorkspaceSize` 中的 `aclnnXX` 部分。如 `aclnnAdd`、`aclnnCat` |
| `tensor_view_shapes` | 张量 shape。如 `"((2,3),(2,3))"`。支持 TensorList 嵌套 |
| `tensor_dtypes` | 张量 dtype。如 `"('float32','float32')"`。支持 TensorList 嵌套。支持单值广播：`"('float16',)"` 表示所有张量使用相同 dtype |

## 完整字段表

### 张量属性

| 字段 | 必填 | 默认 | 说明 |
|------|------|------|------|
| `tensor_formats` | 否 | `('ND',)` | 张量格式。支持嵌套，如 `(('ND','ND'),'ND')`。支持单值广播：`('ND',)` 表示所有张量使用相同 format |
| `tensor_storage_shapes` | 否 | `()` | 非连续张量的存储 shape。回退到视图 shape |
| `tensor_view_offsets` | 否 | `()` | 张量视图在存储中的偏移量 |
| `tensor_view_strides` | 否 | `()` | 张量视图步长。未指定时自动计算 |

### 输出属性

| 字段 | 必填 | 默认 | 说明 |
|------|------|------|------|
| `output_tensor_indexes` | 否 | *(自动)* | 输出张量索引。未设置时自动填充：参数名以 `Ref`/`Out`/`Output`/`OutOptional`/`OutputOptional` 结尾或等于 `output` 的张量识别为输出；Backward/Grad API 会排除 `gradOutput`/`gradOut`/`grad_output`/`attentionOut` 以及非末位的 `output`；无匹配时回退到最后一个张量 |
| `output_inplace_indexes` | 否 | `()` | 原地输出索引（覆写输入）。从 `*Ref` 参数自动填充 |

### 属性与标量

| 字段 | 必填 | 默认 | 说明 |
|------|------|------|------|
| `attributes` | 否 | `{}` | API 属性参数。如 `{'dim': -1}` |
| `scalar_dtypes` | 否 | `()` | 标量参数 dtype。支持 ScalarList 嵌套。**不支持压缩/广播**（标量无 shape，数量只能通过 dtypes 元素数体现），必须逐个显式指定。如 `('int64',)` 或 `(('int32','int32'),)` |
| `scalar_data_ranges` | 否 | `((None,None),)` | 标量数据范围 `(min,max)`。None 时 min=-2、max=2 |
| `my_rank` | 否 | None | HCCL 多卡用例的本卡 rank 序号 |

### Dump 与手动数据

| 字段 | 必填 | 默认 | 说明 |
|------|------|------|------|
| `dump_file_prefix` | 否 | None | Dump 文件名前缀 |
| `manual_tensor_binaries` | 否 | `()` | 手动张量二进制路径。Python 表达式 |
| `manual_golden_binaries` | 否 | `()` | 手动 Golden 二进制路径 |

## 常见模式

### 标量参数

通过 `scalar_dtypes` 指定标量类型。标量值默认随机生成（`scalar_data_ranges` 控制范围），也可在 `attributes` 中指定具体值：

```csv
scalar_dtypes,"('int64',)"
attributes,"{'alpha': 1.0}"
```

随机生成时只需指定 dtype，不需要在 attributes 中填写标量值：

```csv
scalar_dtypes,"('float32',)"
```

### TensorList 嵌套

三层嵌套表示 TensorList 分组：外层括号分组，内层括号是单个张量的 shape。

```csv
tensor_view_shapes,"(((3,3),(3,2)),(3,5))"
```

### 非连续张量

通过 `tensor_view_strides`、`tensor_view_offsets`、`tensor_storage_shapes` 控制张量的非连续布局。视图 shape 和实际存储 shape 可以不同：

```csv
tensor_view_shapes,"((2,3),)"
tensor_view_strides,"((4,1),)"
tensor_view_offsets,"(5,)"
tensor_storage_shapes,"((3,6),)"
```

含义：视图为 2x3，步长 (4,1) 表示行间 stride=4，offset=5 表示从存储的第 5 个元素开始，存储 shape 为 3x6。未指定 strides/offsets/storage_shapes 时默认为连续张量。

### 通过 attributes 指定张量值

在 `attributes` 字典中按接口定义的参数名指定小 shape 张量值，适用于其数值不能由框架随机生成的场景（如索引参数、shape 大小参数、size 参数等，其值与其他输入 shape 有关联）。对于小 shape 张量，比 `manual_tensor_binaries` 或 自定义输入插件 更便捷：

```csv
attributes,"{'group_list': [1, 1, 1]}"
```

也可以单纯为了测试目的指定某个张量的具体值。

## 示例

```csv
testcase_name,api_name,tensor_view_shapes,tensor_dtypes,attributes
aclnn_add_01,aclnnAdd,"((2,3),(2,3))","('float32','float32')","{'alpha': 1.0}"
```

更多示例见 `examples/case_store/aclnn/`。
