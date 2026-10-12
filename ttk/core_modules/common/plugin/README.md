# TestSpec — 算子测试规范

> **不依赖任何测试框架或工具，Ascend 生态统一的算子测试规范。**

## 快速上手

以 `abs` 算子为例，四种测试路径的 spec 注册汇总：

| 测试路径 | 注册名 | 参数名来源 | Golden 参数类型 | 无 spec 时默认行为 |
|---------|--------|----------|----------------|-------------------|
| Kernel | `abs`（CSV `op_name`） | 算子 `def.cpp` 形参 | numpy.ndarray | 回退内置映射（`KERNEL_GOLDEN` → numpy 同名 API，如 `abs`），无命中才报 UNSUPPORTED |
| GEIR | `abs`（CSV `op_name`） | 算子 `def.cpp` 形参 | numpy.ndarray | 回退内置映射（`KERNEL_GOLDEN` → numpy 同名 API，如 `abs`），无命中才报 UNSUPPORTED |
| ACLNN | `aclnnAbs`（CSV `api_name`） | `aclnn_abs.h` 形参 | torch.Tensor | 自动从 torch 同名 API 映射 |
| E2E | `torch.abs`（CSV `api_name`） | torch API 签名 | torch.Tensor | 框架自动在 CPU 跑同 API 作 golden |

> 注册名即 `get_spec_attr` 的查找 key：Kernel/GEIR 用 `op_name`，ACLNN/E2E 用 `api_name`。GEIR 复用 Kernel 的 spec，无需额外编写。类名按 PascalCase+TestSpec 约定或通过 `__spec__` dict 显式注册。

### Kernel

```python
__spec__ = {"abs": "AbsTestSpec"}

import torch

class AbsTestSpec:
    """Abs 算子测试规范（kernel 流程）"""
    def golden(x, **kwargs):
        return [torch.abs(torch.from_numpy(x)).numpy()]

    tolerance = {"float32": {"standard": "binary_equal"}}
```

### ACLNN

```python
__spec__ = {"aclnnAbs": "AclnnAbsTestSpec"}

import torch

class AclnnAbsTestSpec:
    """Abs 算子测试规范（aclnn 流程）"""
    def golden(x, **kwargs):
        return [torch.abs(x)]

    third_party = {"torch": "torch.abs"}
    tolerance = {"float32": {"standard": "binary_equal"}}
```

### E2E

```python
__spec__ = {"torch.abs": "TorchAbsTestSpec"}

import torch

class TorchAbsTestSpec:
    """Abs 算子测试规范（e2e 流程）"""
    def golden(x, **kwargs):
        return [torch.abs(x)]

    tolerance = {"float32": {"standard": "binary_equal"}}
```

## Spec 字段一览

所有字段可选，按需声明：

| 字段 | 用途 | 合法类型 | 适用路径 |
|------|------|---------|---------|
| `golden` | CPU 真值 | `str` / 函数 / 类 | Kernel / GEIR / ACLNN / E2E |
| `third_party` | 三方标杆 | `str` / `dict` / 类 | Kernel / GEIR / ACLNN / E2E |
| `compare` | 自定义比对 | 函数 | Kernel / GEIR / ACLNN / E2E |
| `pre_compare` | 比对前处理 | 函数 | Kernel / GEIR / ACLNN / E2E |
| `customize_inputs` | 自定义输入 | 函数 | Kernel / GEIR / ACLNN / E2E |
| `tolerance` | 精度标准 | `dict(dtype→{standard, ...})` | Kernel / GEIR / ACLNN / E2E |
| `torch_graph` | Graph 模式 | `torch.nn.Module` 子类 | E2E（仅 torch API） |
| `npu_preprocess` | NPU 执行前设备侧预处理钩子 | 函数 | ACLNN / E2E |
| `describe` | 算子描述信息（保留字段，当前框架暂未消费） | 函数 | 全路径 |

所有属性可选。`golden` 三种形式：

| 形式 | 示例 | 适用场景 |
|------|------|---------|
| 字符串 | `golden = "numpy.abs"` | numpy 有直接对应 API |
| 函数 | `def golden(x, *, axis=-1):` | 中等复杂度，`*` 前=输入，`*` 后=属性 |
| 类 | `class Golden: def __call__(self, x):` | 需要状态管理，`__init__`（可选）+ `__call__`（必须） |

> golden 在 CPU 计算，Kernel/GEIR 传入 numpy.ndarray，ACLNN/E2E 传入 CPU（host）侧 torch.Tensor（`torch.from_numpy` 构造，先于设备执行生成；设备侧张量仅 `npu_preprocess` hook 可见）。

`third_party` 三种形式：

| 形式 | 示例 | 适用场景 |
|------|------|---------|
| 字符串 | `third_party = "torch.nn.functional.softmax"` | 单 API 对标 |
| dict | `third_party = {"torch": "...", "tf": "..."}` | 多 vendor / 多框架 |
| 类 | `third_party = MyImplClass` | 小算子拼接，需要 `__call__` |

> third_party 调用三方 API（如 torch），Kernel/GEIR 会将 numpy 转 torch 后调用，ACLNN/E2E 直接传入 torch.Tensor（三方执行侧再上设备）。

### 类形式参数绑定（Impl class）

> `golden` / `third_party` 用类形式时适用。框架按参数名从算子输入/属性池中按名匹配，分别传给 `__init__` 和 `__call__`。

- **严格按名匹配，未命中快速失败**：`__init__` 与 `__call__` 的参数（除 `self`、`*args`、`**kwargs`）按名从算子输入/属性池匹配；按名未命中且无默认值的参数直接抛 `UnknownParamError`（参数名拼错/改名时快速失败，不做位置兜底）。未被具名消费的池条目只能经 `*args`（按池插入顺序，即算子定义参数顺序）或 `**kwargs` 进入。
- **按方法声明分发**：每个输入/属性只传给声明了该参数名的方法；若 `__init__` 和 `__call__` 同时声明同名参数，则两者都收到。
- **带默认值的参数可不传**：算子定义中未出现的参数，若声明了默认值则取默认值（如 `def __call__(self, x, axis=-1)` 中 `axis` 未传时取 -1）。
- **`*args` 收集未消费条目**：当算子有与 Python 保留名冲突的参数（如 ACLNN 接口入参名 `self`），可用 `*args` 按算子定义顺序接收所有未被具名消费的输入/属性，无需逐个声明参数名。`self` 条目不会注入 `**kwargs`（避免 `multiple values for argument 'self'`），只能通过 `*args` 或具名参数（非 `self`）获取。
- **`device` 为 `third_party` 实现类的框架保留参数**：`bind_params` 仅在三方实现（本地/远端 XPU 执行）中注入 `device`；`golden` 类形式不注入——声明 `device` 参数须自带默认值或经 `**kwargs` 获取。有输入 tensor 时设备信息也可直接取 `tensor.device`。

```python
class AddImpl:
    def __init__(self, x):       # x 可进 __init__
        self.x = x
    def __call__(self, x, y):    # 同名 x 两边都喂；y 只喂 __call__
        return [self.x + x + y]

class EyeImpl:                          # third_party 实现类（golden 类形式不注入 device）
    def __init__(self, n, device):    # 无输入 tensor: device 框架注入
        self.n, self.device = n, device
    def __call__(self):
        import torch
        return [torch.eye(self.n, device=self.device)]

# ACLNN 入参名含 self 时，用 *args 按位置接收
class AclnnSelfOpImpl:
    def __call__(self, *args, **kwargs):
        # args[0] 对应 aclnn 接口的 self 张量, args[1] 为 other, ...
        import torch
        return [torch.add(args[0], args[1])]
```

`compare` / `pre_compare` 消费规则：

- ACLNN 必须用 CSV 中的精确 `aclnn*` `api_name` 注册；Kernel 必须用 CSV 中的精确 raw `op_name` 注册。
- Kernel 会对每个已启用的 dynamic、const 和 binary 模式分别调用 hook，关闭模式哨兵不进入 hook。
- 需要读取当前或 replay 恢复输入时，compare 必须显式声明仅关键字参数 `compare_context`；仅声明 `**kwargs` 不会收到该参数。

`customize_inputs` 自定义输入生成：

- 参数名与算子定义一致（按算子/接口签名顺序传入，含标量与输出张量）。
- 若只需调整随机数据范围，可不写此字段，直接在 CSV 中设置 `input_data_ranges`。
- 返回值契约按通路不同：

| 通路 | 入参类型 | 返回值处理 |
|------|---------|-----------|
| Kernel / GEIR | numpy.ndarray | **必须返回**修改后的输入序列（TensorList 可嵌套可平铺，展平后数量须与输入槽一致；只原地改而不返回会报错） |
| ACLNN | torch.Tensor（非 torch 原生 dtype 时为 numpy 视图） | 返回序列按输入张量槽回写（纯输出槽除外，`None` 槽保持原值）；整体返回 `None` 表示原地写法 |
| E2E | torch.Tensor（numpy 回退场景为 ndarray） | 返回值被忽略，**仅原地写法**（`x[:] = value` / `x.copy_(...)`） |

```python
def customize_inputs(x, min_val, max_val, **kwargs):
    if min_val[0] == max_val[0]:
        min_val[0] = numpy.min(x)
        max_val[0] = numpy.max(x)
    return (x, min_val, max_val)   # Kernel/GEIR：返回修改后的输入
```

`tolerance` 精度标准：

按 dtype 声明每个输出的精度比对标准。`standard` 字段只接受官方标准：

| standard 值 | 含义 | `--compare` 传参 |
|-------------|------|------------------|
| `mix_tolerance` | 混合容差（默认，生态算子开源精度标准） | `mixed` |
| `stat_rel_err` | 统计相对误差 | `stat_rel_err` |
| `binary_equal` | 逐 bit 相等 | `binary`（`bin` 为内部判据别名） |
| `cross_check` | 交叉比对（需配合 `third_party`） | `cross_check` |
| `quant` | 量化比对（int4/int8 量化输出，绝对误差 ≤ 1） | —（仅 Spec.tolerance 声明） |
| `isclose` | numpy.isclose 容差比对 | `close` |
| `cosine` | 余弦相似度 | `cosine` |

> 前五行为官方标准，可写入 `Spec.tolerance`；`isclose` 和 `cosine` 为 CLI 框架增强，仅通过 `--compare` 指定。CLI 可用值以 `--compare` 的 choices 为准（`bin` 仅为内部判据别名，不能直接传入）。
> `mix_tolerance` 可用 `rtol`/`atol`/`required_matched_ratio`/`max_abs_error_limit` 覆盖默认阈值表。`max_abs_error_limit` 默认按 `max(兜底值, 32×ULP(最大误差点))` 动态计算；显式配置将完全替代动态计算（配置值即最终上限，不再叠加 ULP——大值域将失去值域自适应放行）。

`torch_graph` Graph 模式（仅 E2E）：

声明一个 `torch.nn.Module` 子类，用于 E2E 的 Graph 模式构图测试。仅适用于 torch API。

```python
import torch.nn as nn

class SoftmaxGraph(nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1)

class TorchSoftmaxTestSpec:
    torch_graph = SoftmaxGraph
```

`describe` 算子描述信息：

返回算子的描述性信息（如约束、备注等），函数形式，返回 str。保留字段，当前框架暂未消费（仅做类型校验）。

## 命名与注册

### 类命名约定

`op_name` → 类名 = `PascalCase + TestSpec`：

| op_name | 类名 |
|---------|------|
| `abs` | `AbsTestSpec` |
| `softmax_v2` | `SoftmaxV2TestSpec` |
| `layer_norm` | `LayerNormTestSpec` |

> 仅处理纯 snake_case（如 `softmax_v2` → `SoftmaxV2`）。对于已是 PascalCase 的算子名（如 `BatchMatMul` → `Batchmatmul`），或期望类名与机械转换结果不一致时（如想要 `Matmul3DTestSpec` 而非 `matmul_3d` 机械转换出的 `Matmul3dTestSpec`），需用 `__spec__` 显式注册。

### `__spec__` 显式注册

```python
# 值为类名字符串（非类对象）；可写在文件顶部
__spec__ = {"abs": "AbsTestSpec"}
```

### 文件与发现规则

- **文件命名**：spec 文件名不以 `_` 开头，否则 loader 跳过该文件（`__init__.py` 例外，不被跳过）。
- **静态索引**：首次加载时，loader 遍历目录下所有 `.py` 文件，通过 `ast.parse` 解析模块级 `__spec__` 赋值，构建 `op → (file, classname)` 索引。此过程仅做语法解析，不执行文件代码，无副作用。
- **惰性执行**：仅当某个 op 被查询命中时，loader 才加载（exec）其所在的 spec 文件并缓存模块对象；未命中的文件在本次会话中不会被执行。
- **`__spec__` 位置不限**：因索引阶段只读 AST 不执行代码，`__spec__` 写在文件顶部或底部效果相同；建议写在顶部，使 op→class 映射一目了然。

## 消费方式

```python
from ttk.core_modules.common.plugin import SpecResolver, get_spec_class_meta

resolver = SpecResolver("softmax_v2", "kernel", ("path/to/specs",))
golden_fn = resolver.resolve_golden()
tp = resolver.resolve_third_party()
meta = get_spec_class_meta("softmax_v2", ("path/to/specs",))   # {spec_file, class_name}
```

## 更多示例

见 `examples/` 目录：

| 文件 | 场景 |
|------|------|
| `01_minimal_golden.py` | 最简 golden（单个函数） |
| `02_golden_three_forms.py` | golden 三种形式：字符串/函数/类 |
| `03_third_party_three_forms.py` | third_party 三种形式：字符串/dict/类 |
| `04_golden_tolerance_compare.py` | golden + tolerance + compare 组合 |
| `05_pre_compare_customize_inputs.py` | pre_compare 两种模式、customize_inputs、多输出 compare |
| `06_golden_multi_path.py` | 同一算子在 Kernel/ACLNN/E2E 不同路径下的 golden 编写 |
