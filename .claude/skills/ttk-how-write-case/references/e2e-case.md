# E2E 模式用例编写

子命令：`python3 -m ttk e2e`
结构类：`TestcaseE2e`（标准字段 23 个：13 公共 + 10 专有；另有 `attributes1`～`attributes9` 扩展列，全量 32 个）

## 用例配置依据

E2E 用例的参数（张量数量/顺序/dtype、关键字参数等）来自框架 API 的函数签名。查看优先级：
1. **PyTorch 官方文档**：https://pytorch.org/docs/stable/ 查找对应 API 的参数定义
2. **torch_npu 扩展文档**：torch_npu 特有 API（如 `torch_npu.npu_conv2d`）查看其扩展文档
3. **源码签名**：直接查看 API 的 Python 定义或 C++ binding

张量参数按 API 签名顺序排列到 `tensor_view_shapes` / `tensor_dtypes`，非张量参数放入 `attributes`。

## API 类型

| api_name 示例 | 类型 | 说明 |
|---------------|------|------|
| `torch.add` | 模块函数 | 直接调用 |
| `torch.nn.functional.relu` | 子模块函数 | 调用子模块中的函数 |
| `torch.Tensor.relu_` | Tensor 方法 | 通过 Tensor 实例调用（原地操作） |
| `torch.ops.cann_ops_transformer.causal_conv1d_fn` | torch.ops 自定义算子 | torch extension 包注册的算子 |

## 必填字段

| 字段 | 说明 |
|------|------|
| `api_name` | 框架 API 路径。如 `torch.add`、`torch.nn.functional.relu`、`torch.Tensor.relu_` |
| `tensor_view_shapes` | 张量 shape，按 API 签名的张量参数顺序排列。如 `"((2,3,4),(2,3,4))"`。支持 TensorList 嵌套 |
| `tensor_dtypes` | 张量 dtype，与 tensor_view_shapes 一一对应。如 `"('float32','float32')"`。支持广播：`"('float32',)"` 填充所有位置 |

## 完整字段表

### 张量属性

| 字段 | 必填 | 默认 | 说明 |
|------|------|------|------|
| `tensor_formats` | 否 | `()` | 张量格式。支持广播 |
| `tensor_storage_shapes` | 否 | `()` | 非连续张量的存储 shape。回退到视图 shape |
| `tensor_view_offsets` | 否 | `()` | 张量视图在存储中的偏移量 |
| `tensor_view_strides` | 否 | `()` | 张量视图步长。未指定时自动计算 |

### 输出与属性

| 字段 | 必填 | 默认 | 说明 |
|------|------|------|------|
| `output_tensor_indexes` | 否 | *(自动)* | 输出张量索引。Inplace Tensor 方法（`torch.Tensor.xxx_`）自动填充 `(0,)`；带 `out` 参数的 API 需手动指定输出位置索引。未设置且 API 无 `out` 参数时，框架自动处理返回值 |
| `inplace_input_indexes` | 否 | `()` | 原地改写输入张量的索引（图模式捕获前备份、重放后还原这些输入；Inplace Tensor 方法自动填充的是 `output_tensor_indexes=(0,)`，此字段需手动声明） |
| `attributes` | 否 | `{}` | 框架 API 关键字参数。如 `{'alpha': 1.0}`。API 签名中的必选非张量参数**必须**提供，否则用例校验不通过 |
| `golden_api` | 否 | `""` | 替代 Golden 计算的 API（详见常见模式） |

## 常见模式

### Inplace Tensor 方法

`torch.Tensor.xxx_` 格式的 API 是原地操作（如 `torch.Tensor.relu_`、`torch.Tensor.add_`）。框架自动将 `output_tensor_indexes` 设为 `(0,)`，无需手动指定：

```csv
testcase_name,api_name,tensor_view_shapes,tensor_dtypes
relu_inplace,torch.Tensor.relu_,"((2,3),)","('float32',)"
```

### 带 out 参数的 API

当 API 签名包含 `out` 参数时（如 `torch.add(..., out=result)`），需要通过 `output_tensor_indexes` 指定输出张量的位置：

```csv
testcase_name,api_name,tensor_view_shapes,tensor_dtypes,attributes,output_tensor_indexes
add_out,torch.add,"((2,3),(2,3),(2,3))","('float32','float32','float32')","{'alpha':1.0}","(2,)"
mm_out,torch.mm,"((2,3),(3,4),(2,4))","('float32','float32','float32')",{},"(2,)"
```

第 3 个张量（索引 2）是 `out` 输出张量。

### golden_api 替代 Golden

当被测 API 与 Golden 计算应使用不同实现时，通过 `golden_api` 指定。TTK 在待测后端调用 `api_name`，在 CPU 上调用 `golden_api` 生成 Golden，然后比对。特殊值 `"disable"` 可禁用 Golden 生成：

```csv
api_name,"torch_npu.npu_conv2d"
golden_api,"torch.nn.functional.conv2d"
```

### 非连续张量

通过 `tensor_view_strides`、`tensor_view_offsets`、`tensor_storage_shapes` 控制张量的非连续布局：

```csv
tensor_view_shapes,"((2,3),)"
tensor_view_strides,"((4,1),)"
tensor_view_offsets,"(5,)"
tensor_storage_shapes,"((3,6),)"
```

未指定 strides/offsets/storage_shapes 时默认为连续张量。

### TensorList 嵌套

三层嵌套表示 TensorList 分组：外层括号分组，内层括号是单个张量的 shape。

```csv
tensor_view_shapes,"(((3,3),(3,2)),(3,5))"
```

## 示例

### 基本示例

```csv
testcase_name,api_name,tensor_view_shapes,tensor_dtypes,attributes
torch_add_01,torch.add,"((2,3),(2,3))","('float32','float32')",
```

### 带属性

```csv
testcase_name,api_name,tensor_view_shapes,tensor_dtypes,attributes
torch_add_02,torch.add,"((2,3),(2,3))","('float32','float32')","{'alpha':1.0}"
```

### 广播 shape

```csv
testcase_name,api_name,tensor_view_shapes,tensor_dtypes,attributes
add_broadcast,torch.add,"((4,1,3),(1,5,3))","('float32','float32')","{'alpha':2.0}"
```

更多示例见 `examples/case_store/e2e/`。
