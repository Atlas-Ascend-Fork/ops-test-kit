# E2E 流程 TestSpec 示例（torch）

E2E 在框架 API 层运行，`api_name` 是框架 API 路径（如 `torch.abs`、`torch.Tensor.relu_`、`torch.ops.cann_ops_transformer.causal_conv1d_fn`）。`golden` / `customize_inputs` 用 **torch.Tensor**（CPU host 侧张量，`torch.from_numpy` 构造，先于设备执行生成）。参数名和顺序与 torch API 签名一致。

注册名 = CSV `api_name`（如 `torch.abs`），类名 `TorchAbsTestSpec`（PascalCase+TestSpec）或 `__spec__` 显式注册。

> [!TIP]
> **默认不需要 golden**：未声明 spec 时框架自动在 CPU 跑同 API 作 golden。只有被测算子无 CPU 等价实现或需要调整比对标准时才写 spec。

## 何时需要写 spec

| 场景 | 资产 |
|------|------|
| 被测算子无 CPU 等价实现（如 `torch_npu` 专属 API） | CSV `golden_api`（改 CPU golden API）或 `golden` |
| torch.ops 自定义算子（无 CPU 实现） | `golden` |
| 控制默认容差 | `tolerance` |
| Graph 模式构图 | `torch_graph` |
| 动态 `-1` 槽在设备侧物化 | `npu_preprocess` |
| 三方标杆对照 | `third_party` |

## golden 示例

```python
import torch

__spec__ = {"torch.abs": "TorchAbsTestSpec"}

class TorchAbsTestSpec:
    def golden(x, **kwargs):
        return [torch.abs(x)]
```

### golden_api（CSV 字段，替代 golden 的简单方案）

当被测 API 与 CPU 参考实现不同名时，直接在 CSV 声明 `golden_api`（TTK 在待测后端调 `api_name`，在 CPU 调 `golden_api` 生成 golden），无需写 golden 函数：

```csv
testcase_name,api_name,golden_api
npu_gelu_f16,torch_npu.npu_gelu,torch.nn.functional.gelu
```

`golden_api="disable"` 禁用 golden 生成（纯执行验证）。

## torch.ops 自定义算子

`api_name` 写 4 段 `torch.ops.<ns>.<op>`（如 `torch.ops.cann_ops_transformer.causal_conv1d_fn`），参数来自算子自带 schema（非 torch API 签名），**无需手写签名**。golden 按 schema 参数顺序接收 torch.Tensor：

```python
class CausalConv1dFnTestSpec:
    def golden(x, weight, bias, seq_idx, **kwargs):
        import torch
        # CPU 参考实现须按算子语义编写；此处仅示意参数接收顺序
        return [x]
```

## customize_inputs（仅原地写法）

E2E 的返回值被忽略，**只能原地修改**（`x[:] = value` / `x.copy_(...)`），返回 `None`：

```python
class TorchAddTestSpec:
    def customize_inputs(x, y, alpha=1.0, **kwargs):
        # 直接修改入参；kwargs 含 testcase_name / tensor_dtypes / input_ranges /
        # batch_applied（batch 一致性片段替换标记）/ testcase 等元信息
        x.copy_(x.clamp(-100, 100))
        return None
```

只需调整随机数据范围时，不写本函数，直接在 CSV 设 `input_data_ranges`。

## torch_graph（Graph 模式构图）

声明 `torch.nn.Module` 子类，仅适用于 torch API：

```python
import torch
import torch.nn as nn

class SoftmaxGraph(nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1)

class TorchSoftmaxTestSpec:
    torch_graph = SoftmaxGraph
```

## third_party（三方标杆）

同 ACLNN：直接收 torch.Tensor（三方执行侧再上设备），dict 多 provider、类形式注入 `device`：

```python
class TorchAddTestSpec:
    third_party = {"torch": "torch.add"}
```

## compare / pre_compare（numpy）

入参**始终为 numpy.ndarray**（任何通路；框架把输出转 numpy 再传入）。需要读取当前或 replay 恢复输入时，必须显式声明仅关键字参数 `compare_context`（仅声明 `**kwargs` 不会收到）。

## tolerance（精度标准）

与 Kernel/ACLNN 相同契约（`mix_tolerance`/`binary_equal`/`isclose`/`stat_rel_err`/`cross_check`/`quant`），示例见 `kernel-plugin.md` 或 `aclnn-plugin.md`。

## npu_preprocess（动态 -1 槽物化）

NPU 执行前（设备锁内、H2D 后、主 API 前）的设备侧预处理钩子。契约：

- 位置参数 = 设备侧输入张量，属性按关键字传入；
- 必须返回 `None`（纯副作用）或 `dict{参数名: 设备张量}`（动态 `-1` 槽物化结果，张量需带 shape/device）；
- 不得声明 `context` 参数（违反抛 `NPU_PREPROCESS_FAILURE`）；
- 自动注入元信息 kwargs（`testcase_name`/`tensor_dtypes`/`input_ranges`/`batch_*` 等，仅在 hook 声明同名参数或 `**kwargs` 时）。

```python
class TorchMyOpTestSpec:
    def npu_preprocess(x, seq_len, **kwargs):
        # x 是设备侧张量；seq_len 按属性关键字传入
        return {"y": torch.zeros((seq_len, 64), device=x.device)}  # 物化 -1 槽
```

## describe（参数化用例组装）

CSV 形状/类型字段任一缺省时触发：只返回 `{"shapes": ..., "dtypes": ...}`，映射到 `tensor_view_shapes`/`tensor_dtypes`。契约同 Kernel（见 `kernel-plugin.md`）。

## 参考

- 详细规范：`docs/reference/test_spec.md`
- 多路径对比示例：`examples/plugin/06_golden_multi_path.py`
- 用例示例：`examples/case_store/e2e/`
