---
name: ttk-how-write-plugin
description: 编写/定义算子测试规范（TestSpec，尚未跑或要新增 golden/输入/精度标准时）。只要用户提到算子测试规范、golden、customize_inputs、自定义输入、third_party、精度标准、tolerance、__spec__、TestSpec、插件、算子没有 golden、怎么写 golden、kernel/aclnn/e2e 的 golden 用 numpy 还是 torch，就必须使用此 skill。即使用户只是说"这个算子没 golden 怎么办"、"golden 用 numpy 还是 torch"、"怎么写 tolerance"，也应触发。已跑完结果不对要排障用 ttk-how-diagnose。
---

# 编写算子测试规范（TestSpec）

算子测试规范（Golden / 自定义输入 / 三方标杆 / 精度标准 / 自定义比对）通过 **TestSpec**（`__spec__` 标记）定义，归属于算子仓库（如 ops-math、ops-nn 的 `tests/assets/`），与测试框架完全解耦。任何工具都可消费，不依赖 TTK。

## 何时需要

| 场景 | 资产 | 说明 |
|------|------|------|
| 算子无内置 Golden | `golden` | 自定义 CPU 参考输出 |
| 需要特殊输入数据 | `customize_inputs` | 自定义输入生成逻辑 |
| 需要三方标杆对照 | `third_party` | torch/tf 等参考实现 |
| 默认比对不适用 | `compare` / `pre_compare` | 自定义比对逻辑 |

## 快速上手

以 `abs` 算子为例，四种测试路径的 spec 注册汇总：

| 测试路径 | 注册名 | 参数名来源 | Golden 参数类型 | 无 spec 时默认行为 |
|---------|--------|----------|----------------|-------------------|
| Kernel | `abs`（CSV `op_name`） | 算子 `def.cpp` 形参 | numpy.ndarray | 回退内置映射（`KERNEL_GOLDEN` → numpy 同名 API，如 `abs`）；无命中才报 UNSUPPORTED |
| GEIR | `abs`（CSV `op_name`） | 算子 `def.cpp` 形参 | numpy.ndarray | 同 Kernel（复用同一回退链） |
| ACLNN | `aclnnAbs`（CSV `api_name`） | `aclnn_abs.h` 形参 | torch.Tensor | 自动从 torch 同名 API 映射 |
| E2E | `torch.abs`（CSV `api_name`） | torch API 签名 | torch.Tensor | 框架自动在 CPU 跑同 API 作 golden |

> 注册名即 `get_spec_attr` 的查找 key：Kernel/GEIR 用 `op_name`，ACLNN/E2E 用 `api_name`。GEIR 复用 Kernel 的 spec，无需额外编写。

> **torch.ops 自定义算子（torch extension 包，如 `cann_ops_transformer`）**：E2E 的 `api_name` 写 4 段 `torch.ops.<ns>.<op>`，参数来自算子自带 schema（非 torch API 签名），**无需手写签名**。详见 `ttk-how-write-case` 的 E2E 引用「torch.ops 自定义算子」。

```python
__spec__ = {"abs": "AbsTestSpec"}

import torch

class AbsTestSpec:
    """Abs 算子测试规范（kernel 流程）"""
    def golden(x, **kwargs):
        return [torch.abs(torch.from_numpy(x)).numpy()]

    tolerance = {"float32": {"standard": "binary_equal"}}
```

## 数据类型

- **`golden`**：在 CPU 计算真值，按测试路径区分入参类型

  | 流程 | 入参类型 | 说明 |
  |------|---------|------|
  | Kernel / GEIR | numpy.ndarray | 需内部 `torch.from_numpy()` 转 torch 计算后 `.numpy()` 转回 |
  | ACLNN / E2E | torch.Tensor | CPU（host）侧张量（`torch.from_numpy` 构造，先于设备执行生成），直接用 torch 计算 |

- **`third_party`**：调用三方 API（如 torch）

  | 流程 | 入参类型 | 说明 |
  |------|---------|------|
  | Kernel / GEIR | torch.Tensor | 框架将 numpy 转 torch 后调用 |
  | ACLNN / E2E | torch.Tensor | 三方执行侧再上设备（远端/本地 XPU 执行时 `.to(device)`） |

- **`compare` / `pre_compare`**：入参**始终为 numpy ndarray**（任何场景；框架把输出转 numpy 再传入）

> 字符串形式（`golden = "torch.abs"`）由框架自动处理 numpy↔torch 转换，所有路径通用。

> E2E 默认由框架在 CPU 跑同 API 作 Golden；如需控制输入范围用 CSV 的 `input_data_ranges` 字段。

## NPU Golden（`--golden-mode NPU`）

golden 在 NPU 上执行并采集性能（msprof）。**device 契约**：声明 `device` 命名参数或 `**kwargs` 的 golden 会收到设备字符串（如 `"npu:0"`），自行 `.to(device)` 上板；框架负责注册 npu 设备类型（golden 只 import torch 也能 `.to(device)`）。

| 流程 | NPU 模式下 golden 收到 | 上板方式 |
|------|----------------------|---------|
| Kernel / GEIR | **numpy.ndarray** + device 字符串 | golden 自行转（如 `torch.as_tensor(x).to(device)`） |
| ACLNN / E2E | **torch.Tensor（已在设备上）** + device 字符串 | 框架已 `.to(device)`，golden 直接算；device 串留给创建新张量 |

- 字符串形式（`golden = "torch.abs"`）：框架直接转张量上板，无需声明 device
- class 形与函数形同契约（kernel 为 numpy 池 + device 注入，aclnn/e2e 为设备张量池）
- numpy 实现的 golden 无法上板：CPU 现算，结果列 `golden_source="cpu"`、性能 `"----"`
- 执行/采集失败自动回落 CPU golden（屏显 ERROR + 根因），精度不受影响
- 结果列：`golden_source`（npu/cpu）、`golden_device_perf_us`、`golden_kernel_count/details`

```python
class AddTestSpec:
    def golden(x1, x2, device=None, **kwargs):   # 声明 device 才收到
        import torch
        t = torch.as_tensor(x1).to(device) + torch.as_tensor(x2).to(device)
        return [t]                                # torch 张量返回，框架转回 numpy
```

## 资产属性

所有属性可选，按需声明：

| 属性 | 用途 | 类型 |
|------|------|------|
| `golden` | CPU 真值 | `str` / 函数 / 类 |
| `customize_inputs` | 自定义输入 | 函数 |
| `third_party` | 三方标杆 | `str` / `dict` / 类 |
| `tolerance` | 精度标准 | `dict(dtype→{standard,...})` |
| `compare` | 自定义比对 | 函数 |
| `pre_compare` | 比对前处理 | 函数 |
| `torch_graph` | Graph 模式 | `torch.nn.Module` 子类（仅 torch API） |
| `npu_preprocess` | NPU 执行前设备侧预处理钩子（-1 动态槽物化） | 函数（ACLNN / E2E） |
| `describe` | 算子描述信息（保留字段，当前未消费） | 函数 |

## golden 三形式

| 形式 | 示例 | 适用 |
|------|------|------|
| 字符串 | `golden = "torch.abs"` | 有直接对应 API，框架自动处理类型转换 |
| 函数 | `def golden(x, *, axis=-1):` | 中等复杂度 |
| 类 | `golden = GoldenCls` | 需要状态管理（`__init__` + `__call__`） |

函数形式两种写法 —— 类内直接定义方法（方法名必须是 `golden`），或独立函数 + 类属性赋值（函数名任意）：

```python
class AbsTestSpec:
    def golden(x, **kwargs):          # 写法1：类内方法
        return [torch.abs(torch.from_numpy(x)).numpy()]

# 写法2：独立函数 + 类属性赋值
def abs_golden(x, **kwargs):
    return [torch.abs(torch.from_numpy(x)).numpy()]

class AbsTestSpec:
    golden = abs_golden
```

> `customize_inputs` / `compare` / `pre_compare` 同理：独立函数需在类中赋值同名属性。

## 函数签名约定

Golden/Input 函数参数分为**输入张量、算子属性、框架元信息**三部分。框架按位置传输入张量、按关键字传属性；当**输入有可选（带默认值）且属性有必选（无默认值）**时，为避免位置参数与关键字属性歧义，用 `*` 分隔（`*` 前是输入，`*` 后是属性）：

```python
# 输入有可选 + 属性有必选 → 用 * 分隔
def op_golden(x, y, bias=None, *, axis, keepdims=False, **kwargs):
    ...
```

其他场景不加 `*`：

```python
# 输入全部必选，属性全部可选 → 不加 *
def op_golden(x, y, alpha=1.0, **kwargs):
    ...

# 无属性 → 不加 *
def op_golden(x, y, **kwargs):
    ...
```

类形式时参数名必须是算子输入/属性名（`__init__`/`__call__` 参数并集 ⊆ `inputs ∪ attrs`）；有默认值的参数可省略。

## customize_inputs（自定义输入）

参数名匹配算子定义，返回修改后的输入（结构与原输入一致）：

```python
# kernel 流程（numpy）
def customize_inputs(x, min_val, max_val, **kwargs):
    if min_val[0] == max_val[0]:
        min_val[0] = numpy.min(x)
        max_val[0] = numpy.max(x)
    return (x, min_val, max_val)
```

## 多输出算子

返回列表，每个元素对应一个输出：

```python
def sort_golden(x, descending=False, **kwargs):
    x_t = torch.from_numpy(x)
    values, indices = torch.sort(x_t, descending=descending)
    return [values.numpy(), indices.numpy()]
```

## tolerance（精度标准）

按 dtype 声明 `standard` 字段，只接受官方标准：

| standard 值 | 含义 | `--compare` 传参 |
|-------------|------|------------------|
| `mix_tolerance` | 混合容差（默认，生态算子开源精度标准） | `mixed` |
| `stat_rel_err` | 统计相对误差 | `stat_rel_err` |
| `binary_equal` | 逐 bit 相等 | `binary`（`bin` 为内部判据别名，不能直接传给 `--compare`） |
| `cross_check` | 交叉比对（需配合 `third_party`） | `cross_check` |
| `quant` | 量化比对（int4/int8 量化输出，绝对误差 ≤ 1） | —（仅 Spec.tolerance 声明） |
| `isclose` | numpy.isclose 容差比对 | `close` |
| `cosine` | 余弦相似度 | `cosine` |

> 前五行可写入 `Spec.tolerance`；`isclose` 和 `cosine` 为 CLI 框架增强，仅通过 `--compare` 指定。
> `mix_tolerance` 可用 `rtol`/`atol`/`required_matched_ratio`/`max_abs_error_limit` 覆盖默认阈值表（`max_abs_error_limit` 配置后完全替代 ULP 动态上限，配置值即最终上限）。

```python
tolerance = {
    "float32": {"standard": "mix_tolerance"},
    "int8": {"standard": "binary_equal"},
}
```

## 注册与发现

- **类命名**：`op_name` → `PascalCase + TestSpec`（如 `abs` → `AbsTestSpec`、`softmax_v2` → `SoftmaxV2TestSpec`）
- **`__spec__` dict**：算子名非纯 snake_case 时（如 `BatchMatMul`）用 `__spec__ = {"op": "ClassName"}` 显式注册（值为类名字符串；loader AST 扫描不 exec 文件）
- **文件命名**：spec 文件不能以 `_` 开头（loader 跳过）
- **惰性加载**：loader 首次按 `__spec__` AST 建索引，仅命中时 exec 该文件

## 使用

`--plugin` 指定 spec 文件或目录，建议使用绝对路径。指向目录时框架逐层扫描：

```shell
# 指向目录（算子仓通常放 tests/assets/）
python3 -m ttk kernel -i cases.csv --plugin /path/to/ops-math/math/abs/tests/assets/

# 多个路径（逗号分隔）
python3 -m ttk aclnn -i cases.csv --plugin /path/to/assets_a/,/path/to/assets_b/
```

## 规则

1. **注册名必须匹配**：类名（`PascalCase+TestSpec`）或 `__spec__` 的 key 必须与 CSV 的 `op_name` / `api_name` 完全一致
2. **参数顺序**：函数参数名和顺序与算子定义文件（def.cpp / aclnn*.h）中的输入参数一致
3. **kwargs 始终接收**：通过 `**kwargs` 接收元信息（dtypes、shapes、formats、soc_version 等，完整字段见 `references/kernel-plugin.md` / `aclnn-plugin.md`）
4. **返回类型**：`golden` 返回列表（每个输出一个元素）；`customize_inputs` 返回与输入结构一致的 tuple
5. **禁止 `register()` 手写签名**：不要在 golden 文件里 `from ttk...import ParamInfo/APIParamInfo/FrameworkApiInfoKeeper` 再调 `register()` 声明算子签名——签名应来自算子自身（E2E 的 `torch.ops` 算子从 `_schemas` 自动解析；Kernel/GEIR 从 `def.cpp`；ACLNN 从 `aclnn*.h`）。手写 `register()` 会把 golden 耦合到 ttk 内部类（ttk 改内部类时所有此类 golden 失效），且与算子真实签名漂移。`register()` 仅在算子**无 schema**（裸 `impl`、未 `define`）时作兜底。

## 调试

```shell
# 检查 spec 文件语法
python3 -c "import ast; ast.parse(open('spec.py').read()); print('ok')"

# 单用例测试
python3 -m ttk kernel -i cases.csv -t case_name --plugin /path/to/assets/
```

常见问题：
- 资产不生效：检查 `--plugin` 路径、类名（`PascalCase+TestSpec`）或 `__spec__` 注册名是否与 CSV `op_name`/`api_name` 一致
- 参数数量不匹配：核对函数签名与算子定义文件的输入参数是否一致

## CSV 数据范围替代

如果只需要调整随机数据范围，可以不写 `customize_inputs`，直接在 CSV 中设置：

```csv
input_data_ranges,"((-1, 1), (0.1, 10))"
```

或使用 `--input-dist normal` 切换为正态分布。

## 详细规范

完整规范见 `ttk/core_modules/common/plugin/README.md`，渐进示例见 `ttk/core_modules/common/plugin/examples/`：

| 文件 | 场景 |
|------|------|
| `01_minimal_golden.py` | 最简 golden（单个函数） |
| `02_golden_three_forms.py` | golden 三种形式：字符串/函数/类 |
| `03_third_party_three_forms.py` | third_party 三种形式：字符串/dict/类 |
| `04_golden_tolerance_compare.py` | golden + tolerance + compare 组合 |
| `05_pre_compare_customize_inputs.py` | pre_compare 两种模式、customize_inputs、多输出 compare |
| `06_golden_multi_path.py` | 同一算子在 Kernel/ACLNN/E2E 不同路径下的 golden 编写 |

按流程的示例与 kwargs 字段：
- kernel 流程（numpy）→ `references/kernel-plugin.md`
- aclnn 流程（torch）→ `references/aclnn-plugin.md`

## 相关 Skill

- 资产写好了怎么跑？→ `ttk-how-run-test`
- 跑不过怎么排查？→ `ttk-how-diagnose`
- 用例 CSV 怎么写？→ `ttk-how-write-case`
