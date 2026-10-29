# TestSpec — 算子测试规范

## 使用须知

TestSpec 是算子测试规范（Golden / 自定义输入 / 三方标杆 / 精度标准 / 自定义比对），归属于算子仓库（如 ops-math 的 `tests/assets/`），不依赖任何测试框架。

以 `abs` 为例，四通路的注册与默认行为：

| 通路 | 注册名 | Golden 参数类型 | 无 spec 时默认行为 |
|------|--------|----------------|-------------------|
| Kernel / GEIR | `abs`（CSV `op_name`） | numpy.ndarray | 回退内置映射（`KERNEL_GOLDEN` → numpy 同名 API） |
| ACLNN | `aclnnAbs`（CSV `api_name`） | torch.Tensor | 自动从 torch 同名 API 映射 |
| E2E | `torch.abs`（CSV `api_name`） | torch.Tensor | 框架自动在 CPU 跑同 API 作 golden |
| E2E onnx | `onnx.Abs`（CSV `api_name`） | numpy.ndarray | onnx ReferenceEvaluator 跑同模型 |

注册名：Kernel/GEIR 用 `op_name`，ACLNN/E2E 用 `api_name`；GEIR 复用 Kernel 的 spec。类名按 `PascalCase+TestSpec` 约定，或用 `__spec__` 显式注册。

```python
__spec__ = {"abs": "AbsTestSpec"}

class AbsTestSpec:
    def golden(x, **kwargs):
        import torch
        return [torch.abs(torch.from_numpy(x)).numpy()]

    tolerance = {"float32": {"standard": "binary_equal"}}
```

## 文档导览

1. **[字段一览](#一字段一览)**
2. **[describe](#二describe)**
3. **[golden 与 third_party](#三golden与third_party)**
4. **[自定义输入与比对](#四自定义输入与比对)**
5. **[tolerance](#五tolerance)**
6. **[其他字段](#六其他字段)**
7. **[命名与注册](#七命名与注册)**
8. **[更多示例](#八更多示例)**

## 一、字段一览

所有字段可选，按需声明：

| 字段 | 用途 | 合法类型 | 适用路径 |
|------|------|---------|---------|
| `describe` | 参数化用例组装 | 函数 | 全通路 |
| `golden` | CPU 真值 | `str` / 函数 / 类 | 全通路 |
| `third_party` | 三方标杆 | `str` / `dict` / 类 | 全通路 |
| `compare` / `pre_compare` | 自定义比对 / 比对前处理 | 函数 | 全通路 |
| `customize_inputs` | 自定义输入 | 函数 | 全通路 |
| `tolerance` | 精度标准 | `dict(dtype→{standard, ...})` | 全通路 |
| `torch_graph` | Graph 模式 | `torch.nn.Module` 子类 | E2E（仅 torch） |
| `npu_preprocess` | 执行前设备侧预处理 | 函数 | ACLNN / E2E |

## 二、describe

CSV 形状/类型字段任一缺省时触发：只返回 `{"shapes": ..., "dtypes": ...}`，框架按通路映射回填；其余字段仍由 CSV 提供。

```python
class GemmTestSpec:
    @staticmethod
    def describe(params):
        m, k, n = params["M"], params["K"], params["N"]
        return {"shapes": ((m, k), (k, n)), "dtypes": ("float16", "float16")}
```

## 三、golden 与 third_party

`golden` 三种形式：

| 形式 | 示例 | 适用 |
|------|------|------|
| 字符串 | `golden = "numpy.abs"` | numpy 有对应 API |
| 函数 | `def golden(x, *, axis=-1):` | 中等复杂度 |
| 类 | `class Golden: def __call__(self, x):` | 需要状态 |

> golden 在 CPU 计算：Kernel/GEIR 与 E2E onnx 传入 numpy.ndarray；ACLNN/E2E（torch/tf）传入 CPU 侧 torch.Tensor。

`third_party` 四种形式：

| 形式 | 示例 | 适用 |
|------|------|------|
| 字符串 | `"torch.nn.functional.softmax"` | 单 API 对标 |
| dict | `{"torch": "...", "tf": "..."}` | 多框架 |
| 函数 | `def third_party(x, *, axis=-1):` | 自定义实现 |
| 类 | `third_party = MyImplClass` | 小算子拼接 |

> third_party 调用三方 API：Kernel/GEIR 传入前把 numpy 转 torch；ACLNN/E2E 直接传 torch.Tensor。

### 类形式写法

`golden` / `third_party` 用类形式时，框架负责实例化并调用，`__init__` 与 `__call__` 需要的算子输入、属性由框架按参数名自动传入：

- 参数名与算子输入名或属性名一致 → 收到对应值；无对应值但有默认值 → 用默认值；无对应值且无默认值 → 报错
- 同一参数名在 `__init__` 与 `__call__` 都声明 → 两处都收到；只在一处声明 → 仅该处收到
- 算子输入名恰好是 Python 保留字（如 aclnnReal 的输入叫 `self`）时无法按名传参，改用 `*args` 按算子定义顺序接收
- `device` 只给 `third_party` 类形式（声明了 `device` 参数才会收到）；`golden` 类形式不传，需要时从输入 tensor 取 `.device`

```python
class AddImpl:
    def __init__(self, x):
        self.x = x
    def __call__(self, x, y):
        return [self.x + x + y]

class EyeImpl:                        # third_party 实现类
    def __init__(self, n, device):    # 无输入 tensor 时 device 由框架注入
        self.n, self.device = n, device
    def __call__(self):
        import torch
        return [torch.eye(self.n, device=self.device)]
```

## 四、自定义输入与比对

### customize_inputs

在随机输入生成与 `attributes` 覆写之后、Golden 生成之前调用，按算子签名顺序传入（含标量与输出张量）。

| 通路 | 入参类型 | 返回值 |
|------|---------|--------|
| Kernel / GEIR | numpy.ndarray | 必须返回修改后的输入序列 |
| ACLNN | torch.Tensor | 返回序列回写输入槽；返回 `None` = 原地写法 |
| E2E | torch.Tensor（onnx 为 ndarray） | 忽略返回值，仅原地写法 |

只调数据范围时直接在 CSV 设 `input_data_ranges`，不用写本字段。

### compare / pre_compare

- 入参始终是 numpy ndarray
- 需要读当前/replay 输入时，显式声明 `compare_context`（只写 `**kwargs` 收不到）

## 五、tolerance

按 dtype 声明每个输出的比对标准：

| standard 值 | 含义 | `--compare` |
|-------------|------|-------------|
| `mix_tolerance` | 混合容差（默认） | `mixed` |
| `stat_rel_err` | 统计相对误差 | `stat_rel_err` |
| `binary_equal` | 逐 bit 相等 | `binary` |
| `cross_check` | 交叉比对（配合 `third_party`） | `cross_check` |
| `quant` | 量化比对（int4/int8 输出） | 仅 Spec 声明 |
| `isclose` | numpy.isclose 容差 | `close` |
| `cosine` | 余弦相似度 | `cosine` |

> [!NOTE]
> - `cosine` 仅能通过 CLI 指定
> - `mix_tolerance` 可用 `rtol` / `atol` / `required_matched_ratio` / `max_abs_error_limit` 覆盖默认阈值
> - `isclose` 可用 `rtol` / `atol` / `ptol` 覆盖

```python
tolerance = {
    "float32": {"standard": "mix_tolerance", "rtol": 0.004},
    "float16": {"standard": "isclose", "rtol": 0.005, "ptol": 0.005},
    "int8": {"standard": "quant", "ptol": 0.001},
}
```

## 六、其他字段

### torch_graph（仅 E2E）

```python
import torch.nn as nn

class SoftmaxGraph(nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1)

class TorchSoftmaxTestSpec:
    torch_graph = SoftmaxGraph
```

### npu_preprocess（ACLNN / E2E）

NPU 执行前（数据拷入设备后）的设备侧钩子：位置参数为设备侧输入张量，属性按关键字传入；返回 `None`（纯副作用）或 `dict{参数名: 设备张量}`（用于 `-1` 动态 shape）。

## 七、命名与注册

`op_name` → 类名 = `PascalCase + TestSpec`：

| op_name | 类名 |
|---------|------|
| `abs` | `AbsTestSpec` |
| `softmax_v2` | `SoftmaxV2TestSpec` |

已是 PascalCase 的算子名或想要自定义类名时，用 `__spec__` 显式注册：

```python
__spec__ = {"abs": "AbsTestSpec"}   # 值为类名字符串
```

文件规则：spec 文件名不能以 `_` 开头（否则被跳过）；`__spec__` 建议写在文件顶部。

## 八、更多示例

见 [`examples/plugin/`](../../examples/plugin/)：

| 文件 | 场景 |
|------|------|
| `01_minimal_golden.py` | 最简 golden |
| `02_golden_three_forms.py` | golden 三种形式 |
| `03_third_party_forms.py` | third_party 字符串 / dict / 函数 / 类形式 |
| `04_golden_tolerance_compare.py` | golden + tolerance + compare |
| `05_pre_compare_customize_inputs.py` | pre_compare / customize_inputs / 多输出 compare |
| `06_golden_multi_path.py` | 同一算子多路径 golden |
