# 精度比对方法

通过 `--compare` 参数选择精度比对方法，或由 TestSpec 的 `tolerance` 字段按 dtype 逐输出路由。不同方法适用于不同的算子类型和精度要求。

## 方法总览

| 方法 | 参数值 | 公式核心 | 适用场景 | 默认容差 |
|------|--------|---------|---------|---------|
| 混合容差 | `mixed`（默认） | `\|a-g\| <= atol + rtol*\|g\|` 逐元素 + 通过率 | 浮点算子常规测试（生态算子开源精度标准） | 按 dtype 阈值表 |
| 统计相对误差 | `stat_rel_err` | `\|a-g\| / (\|g\|+ε)` 的均值/最大值 | 统计误差均值/最大值 | 按 dtype 阈值表 |
| 数值近似 | `close` | `\|a-g\| <= atol + rtol*\|g\|` | 逐点 isclose 比对 | 按 dtype 表（fp16 系 0.001，fp8 系 0.25/0.5，其余浮点 0.0001），atol=1e-8 |
| 余弦相似度 | `cosine` | `dot(a,g) / (\|a\|*\|g\|)` | 大规模向量整体趋势 | 1-cos ≤ 0.01 |
| 二进制精确 | `binary` | 逐字节比对（字节完全一致） | 整型运算/精确结果 | 无容差 |
| 重量化 | `requant` | ULP 差 ≤ 1 | hifloat8 类型 | ptol=0.001 |
| 量化 | `quant` | `\|a-g\| <= 1`（1 LSB） | 浮点输入 + int4/int8 量化输出 | ptol=0 |
| 三方交叉校验 | `cross_check` | NPU/XPU 误差比值 | 多硬件精度对齐 | 按 level 预设 |

> `a` = NPU 输出，`g` = Golden

## 1. 混合容差（mixed）

默认方法，生态算子开源精度标准（混合容差）。结合绝对容差（atol）和相对容差（rtol）逐元素比对。

### 公式

```
元素通过：|a - g| <= atol + rtol * |g|
matched_ratio = 通过元素数 / 总元素数
passed = matched_ratio >= 0.99 AND max_abs_error <= max_abs_error_limit
```

- `atol`：绝对容差，保证小值（golden 接近 0）场景的合理误差，天然避免除零
- `rtol`：相对容差，保证大值场景的相对精度
- `max_abs_error_limit`：绝对误差硬上限，拦灾难性离群点。默认按精度标准动态计算：
  `max(兜底值, 32 × ULP(g_low))`——`g_low` 为最大绝对误差点的 golden 值按算子实际
  输出 dtype 收窄后的值，上限随被测 dtype 与该点值域自适应缩放（标准定义见
  cann/opbase `docs/zh/ops_precision_standard/mixed_tolerance_standard.md` §2.1.2）。
  Spec 中显式配置该参数时将**完全替代**动态计算：配置值即为最终上限，不再叠加
  `32 × ULP` 动态部分——大值域用例将失去值域自适应放行（固定常数偏小则误杀），
  小值域用例将失去兜底收紧以外的下限保护。未配置时按上述标准动态式计算。

### dtype 阈值表

| dtype | rtol | atol | max_abs_error_limit |
|-------|------|------|---------------------|
| `float16` | 2^-9 (≈1.95e-3) | 2^-9 (≈1.95e-3) | 兜底 1e-1，动态 32\*ULP |
| `bfloat16` | 2^-6 (≈1.56e-2) | 2^-6 (≈1.56e-2) | 兜底 1e-0，动态 32\*ULP |
| `float32` | 2^-10 (≈9.77e-4) | 2^-16 (≈1.53e-5) | 兜底 1e-2，动态 32\*ULP |
| `hifloat32` | 2^-9 (≈1.95e-3) | 2^-10 (≈9.77e-4) | 兜底 1e-1，动态 32\*ULP |
| `float8_e4m3fn` | 2^-2 (0.25) | 2^-4 (0.0625) | 兜底 1e-0，动态 32\*ULP |
| `float8_e5m2` | 2^-1 (0.5) | 2^-3 (0.125) | 兜底 1e-1，动态 32\*ULP |
| 其他浮点（回退到 float32 档） | 2^-10 | 2^-16 | 兜底 1e-2，动态 32\*ULP |

> 表中"兜底 + 动态"取两者较大值；ULP 锚点为最大绝对误差点的 golden 收窄值（§上文动态计算说明）。

`required_matched_ratio` 全 dtype 默认值为 `0.99`（Spec 可用同名参数覆盖）。`hifloat8` 输出自动路由到[重量化（requant）](#6-重量化requant)判据，不适用本表。

### 特殊处理

- 双方同位置均为 NaN → 视为一致
- 双方同位置均为 Inf 且同号 → 视为一致
- NaN/Inf 位置不一致 → 视为无界误差，必超硬上限直接 FAIL

### 使能方式

```bash
# CLI（显式指定）
python3 -m ttk kernel -i cases.csv --compare mixed

# Spec.tolerance（按 dtype 路由，需 --plugin；standard 用 mix_tolerance，可覆盖阈值）
# class AddTestSpec:
#     tolerance = {"float32": {"standard": "mix_tolerance", "rtol": 0.002}}
# 未指定 --compare 且无 Spec.tolerance 时自动生效
```

## 2. 统计相对误差（stat_rel_err）

按 dtype 统计相对误差的均值（mere）和最大值（mare），结合社区标准阈值判定。

### 公式

```
rel_err = |a - g| / (|g| + 1e-7)
mere = mean(rel_err)       # 平均相对误差
mare = max(rel_err)        # 最大相对误差
```

### 判定

```
passed = mere < threshold AND mare < 10 * threshold
```

- `mere` 超过 `threshold` → FAIL
- `mare` 超过 `10 * threshold` → FAIL（防止个别大误差被均值掩盖）

### dtype 阈值表

| dtype | threshold | 说明 |
|-------|-----------|------|
| `float16` | 2^-10 (≈0.000977) | 半精度 |
| `bfloat16` | 2^-7 (≈0.007812) | BF16 精度较低，阈值放宽 |
| `float32` | 2^-13 (≈0.000122) | 单精度 |
| `float8_e4m3fn` | 2^-3 (≈0.125) | FP8 E4M3 |
| `float8_e5m2` | 2^-2 (≈0.25) | FP8 E5M2 |
| 其他 | 2^-13 | 回退到 float32 级 |

### 特殊处理

- NaN/Inf 位置不一致 → 直接 FAIL（不计算 mere/mare）
- 双方同位置均为 NaN → 视为一致
- 双方同位置均为 Inf 且同号 → 视为一致

### 使能方式

```bash
# CLI（显式指定）
python3 -m ttk kernel -i cases.csv --compare stat_rel_err

# Spec.tolerance（按 dtype 路由，需 --plugin）
# 需显式指定（--compare 或 Spec.tolerance）；否则默认走 mixed
```

## 3. 数值近似（close）

使用与 `np.isclose(..., equal_nan=True)` 语义等价的逐元素比对（有限位按 rtol/atol 判，同号 inf 相等、双 NaN 视为相等；fp16 先升 fp32 再计算），支持相对误差（rtol）和绝对误差（atol）双判据。

### 公式

```
passed = |a - g| <= atol + rtol * |g|
```

### 默认容差

| dtype | rtol | ptol | atol |
|-------|------|------|------|
| `float16` / `bfloat16` / `complex32` | 0.001 | 0.001 | 1e-8 |
| `float8_e4m3fn` | 0.25 | 0.25 | 1e-8 |
| `float8_e5m2` | 0.5 | 0.5 | 1e-8 |
| 其他浮点（如 `float32`/`float64`） | 0.0001 | 0.0001 | 1e-8 |

- `rtol`：相对误差容差（大值域）
- `atol`：绝对误差容差（小值域，接近 0 时 rtol 失效）
- `ptol`：允许的不匹配元素比例（如 0.001 = 允许 0.1% 元素超出容差）

### 判定

```
precision = (golden_size - mismatch_count) / golden_size
passed = (1 - precision) <= ptol
```

### CSV 自定义容差

```csv
precision_tolerances,"((0.001, 0.001),)"
absolute_precision,1e-8
```

- `precision_tolerances`：每个输出的 `(rtol, ptol)` 对
- `absolute_precision`：atol。单个浮点数应用于全部输出；tuple/list 时按输出索引逐输出取值，多输出算子各输出量纲不同时可分别设置：

```csv
absolute_precision,"(1e-8, 1e-5)"
```

> 字段类型为 `FLOAT_SCALAR_NESTED`，完整字段约定见[测试用例生成](./Test_Case_Generation.md)。

### 使能方式

```bash
# CLI
python3 -m ttk kernel -i cases.csv --compare close

# Spec.tolerance（rtol/atol/ptol 可覆盖 dtype 分档默认；legacy CSV 容差仍优先）
tolerance = {"float16": {"standard": "isclose", "rtol": 0.005, "atol": 0.000025, "ptol": 0.005}}
```

## 4. 余弦相似度（cosine）

计算输出与 Golden 的余弦相似度，衡量向量方向一致性而非数值精度。

### 公式

```
cosine = dot(a, g) / (norm(a) * norm(g))
passed = (1 - cosine) <= rtol
```

### 默认容差

| 参数 | 默认值 |
|------|--------|
| `rtol` | 0.01（即 cosine ≥ 0.99） |

### 适用场景

- 大规模向量（如 embedding、attention 输出）
- 关注整体趋势而非逐点精度
- 不适合需要精确数值的场景

### 使能方式

```bash
# CLI
python3 -m ttk kernel -i cases.csv --compare cosine

# Spec.tolerance 不支持 cosine（CLI 框架增强，仅通过 --compare 指定）
```

## 5. 二进制精确（binary）

逐 bit 比对 NPU 输出与 Golden，无容差概念。

### 判定

```
output_bytes == golden_bytes（原生 dtype 逐字节快路径；bfloat16/int4/fp8 等自定义窄 dtype 走分块 SHA-256，判定语义一致）→ PASS
```

不一致时，按 `itemsize` 分组定位差异元素索引。双方同位置均为 NaN 时视为相等。

### 适用场景

| 场景 | 说明 |
|------|------|
| 纯搬移算子 | 输出应与输入逐 bit 一致 |
| 纯比较算子 | 如 Equal、Less、Greater 等 |
| 整型运算 | int8/int32/uint 等精确结果 |
| 索引/掩码输出 | 非 0 即 1 的场景 |
| float4 / int4 | 打包格式逐 bit 比对 |

### 跨 dtype 处理

仅支持整型/布尔间的跨 dtype 比对（自动提升到公共类型）。浮点 vs 整型等不兼容组合直接拒绝。

### 使能方式

```bash
# CLI
python3 -m ttk kernel -i cases.csv --compare binary

# Spec.tolerance（需 --plugin；standard 用 binary_equal）
# class AddTestSpec:
#     tolerance = {"float16": {"standard": "binary_equal"}}
```

## 6. 重量化（requant）

为 hifloat8 类型设计，以 ULP（Unit in Last Place）为判据。float8（e5m2/e4m3fn）默认走 mixed，也可通过 `--compare requant` 显式选择本判据。

### 公式

```
diff = |view_as_int8(a) - view_as_int8(g)|    # FP8/hifloat8 按 int8 视图比较 ULP 差
passed = (diff <= 1 的元素占比) >= (1 - ptol)
```

> 非 ULP 视图 dtype（fp16/fp32/bf16 等）显式选择 requant 时，退化为绝对值判据：`|a - g| <= 1`（默认 ptol=0，即全部元素须满足）。

### 默认容差

| dtype | ptol |
|-------|------|
| `hifloat8`（自动路由） | 0.001（允许 0.1% 元素 ULP 差 > 1） |
| `float8_e5m2` / `float8_e4m3fn`（显式 `--compare requant`） | 0.001（允许 0.1% 元素 ULP 差 > 1） |

### 适用场景

- hifloat8 量化算子（自动路由）
- float8（e5m2/e4m3fn）显式选择 ULP 判据时

### 使能方式

```bash
# CLI
python3 -m ttk kernel -i cases.csv --compare requant
```

> `hifloat8` 会自动切换到 `requant`；`float8_e5m2` / `float8_e4m3fn` 默认走 `mixed`（生态算子开源精度标准），也可通过 `--compare requant` 显式选择 ULP 判据。

## 7. 量化（quant）

用于浮点输入 + int4/int8 量化输出的算子，判据为绝对误差 ≤ 1 LSB。

### 公式

```
diff = |int(a) - int(g)|
bad = count(diff > 1)
passed = (bad / golden_size) <= ptol
```

### 默认容差

| 参数 | 默认值 |
|------|--------|
| `ptol` | 0（不允许任何元素超差） |

### 适用场景

- 量化算子（如 PerTokenQuant）
- 浮点输入 → int4/int8 输出
- 量化末步 `round(x / scale)` 是阶跃函数，浮点末位 1 ULP 差异会导致结果跳到相邻整数格，逐位相等既不可能也无意义

### 使能方式

只能通过 Spec.tolerance 声明，不通过 `--compare` 指定。声明时会校验输出 dtype 为 int4/int8 且输入至少含一个浮点（否则无法构成浮点→整型量化）：

```python
__spec__ = {"per_token_quant": "PerTokenQuantTestSpec"}


class PerTokenQuantTestSpec:
    tolerance = {
        "int8": {"standard": "quant", "ptol": 0.001},
    }
```

## 8. 三方交叉校验（cross_check）

将 NPU 输出、Golden、第三方（XPU）输出三方做误差比值比对。完整使用方法详见 [XPU 三方交叉校验与性能采集](./XPU_Cross_Check.md)。

### 公式

```
rel_npu   = |a - g| / (|g| + ε)          # NPU 相对误差
rel_party = |b - g| / (|g| + ε)          # XPU 相对误差

mare_ratio = max(rel_npu) / max(max(rel_party), small_value)   # 最大误差比
mere_ratio = mean(rel_npu) / max(mean(rel_party), small_value) # 平均误差比
rmse_ratio = rmse(a,g) / max(rmse(b,g), small_value)           # RMSE 比
```

> 三比值的分母一律以 `small_value`（按 dtype 取的小值域阈值）夹底：既防除零，也保证竞品误差落在噪声地板以下时不放大比值。NaN 分子 → 比值为 inf；分子分母同号溢出或同为 0 → 比值取 1。

### 判定

大值域（`|g| ≥ small_value`）：三个比值均不超过 limit → PASS

小值域（`|g| < small_value`）：NPU 误差计数 / max(XPU 误差计数, 1) ≤ 2.0 → PASS

### 特殊值处理（NaN/Inf）

特殊位满足任一规则即通过：NPU 与 third_party 一致，或 NPU 与 golden 一致（同为 NaN/±Inf，或同为有限且数值相等）。仅当 NPU 与 golden、third_party 双方均不一致时才判 FAIL——即「NPU 偏离 golden 且竞品能对上 golden」。竞品单方面产出 NaN/Inf（NPU 与 golden 一致）不判失败。

### Level 预设

| Level | mare_ratio | mere_ratio | rmse_ratio | 适用 |
|-------|-----------|-----------|-----------|------|
| L0 | 10.0 | 2.0 | 2.0 | 宽松（初期对齐） |
| L1 | 5.0 | 1.5 | 1.5 | 默认 |
| L2 | 2.0 | 1.2 | 1.2 | 严格（接近对齐） |

### small_value 阈值

| dtype | small_value | small_value_atol |
|-------|------------|-----------------|
| `float16` | 2^-11 | 2^-16 |
| `bfloat16` | 2^-8 | 2^-16 |
| `float32` | 2^-14 | 2^-30 |

> `cross_check` 仅支持 `float16` / `bfloat16` / `float32`。

### 使能方式

```bash
# CLI
python3 -m ttk kernel -i cases.csv --compare cross_check --config ttk.conf.yaml --plugin /path/to/assets.py

# Spec.tolerance（声明 level 和 ratio）
# class AddTestSpec:
#     tolerance = {"bfloat16": {"standard": "cross_check", "level": "L2"}}
```

> 完整使用方法详见 [XPU 三方交叉校验与性能采集](./XPU_Cross_Check.md)。

## 自动切换规则

部分 dtype 会自动切换比对方法，无需手动指定：

| 数据类型 | 自动切换到 | 原因 |
|---------|-----------|------|
| `hifloat8` | `requant` | 需 ULP 判据 |
| `float4` / `int4` | `binary` | 打包格式需逐 bit |
| `float8_e8m0` | `binary` | 无浮点语义 |
| `complex32` / `complex64` / `complex128` | `close`（isclose） | 复数需逐元素 close 判据 |
| 整型 / 布尔 | `binary` | 精确结果 |

## 优先级与路由

每个输出的比对方法按以下优先级决定：

```
1. CLI --compare 参数（最高优先级）
2. Spec.tolerance[dtype].standard（需 --plugin）
3. dtype 自动切换规则
4. mixed（默认兜底）
```

> dtype 类别限制：整型/布尔及 `float4`/`float8_e8m0` 输出强制路由 `binary`（CLI 与 Spec 声明均不生效；整型中仅 int4/int8 经 Spec 显式声明 `quant` 可突破，见下）；complex 输出仅接受 `binary` 族/close 类判据；`hifloat8` 仅接受 `binary`/`requant`。CLI `--compare` 仅对浮点族输出无条件生效。

`quant` 例外：只能通过 Spec.tolerance 声明（CLI 无 `quant` 取值）。Spec 声明 quant 后，`--compare` 的整数适用取值 `binary`/`requant` 仍可覆盖该声明；其余浮点判据（close/cosine/stat_rel_err/mixed/cross_check）不覆盖，维持 quant。

## 容差自定义

### CSV 字段（legacy）

| 字段 | 作用 | 格式 | 读取方 |
|------|------|------|--------|
| `precision_tolerances` | 每输出的 `(rtol, ptol)` | `((0.001, 0.001),)` | `close`、`cosine`（仅 rtol）、`requant`/`quant`（仅 ptol，作为允许超差元素占比） |
| `absolute_precision` | atol；单值应用于全部输出，tuple/list 按输出索引逐输出配置 | `1e-8` / `(1e-8, 1e-5)` | 仅 `close` |

> 默认 `mixed` 不读 CSV 字段，需在 TestSpec `tolerance` 中覆盖（需 `--plugin`）。

### Spec.tolerance

```python
__spec__ = {"add": "AddTestSpec"}


class AddTestSpec:
    tolerance = {
        "float16": {"standard": "mix_tolerance", "rtol": 0.004},
        "float32": {"standard": "stat_rel_err", "threshold": 0.002},
        "bfloat16": {"standard": "cross_check", "level": "L2"},
        "int8": {"standard": "quant", "ptol": 0.001},
    }
```

## 方法选择建议

| 场景 | 推荐方法 | 参数 |
|------|---------|------|
| 浮点算子常规测试 | 混合容差（生态算子开源精度标准） | `--compare mixed`（默认） |
| 统计误差均值/最大值 | 统计相对误差 | `--compare stat_rel_err` |
| 逐点 isclose | 数值近似 | `--compare close` |
| 大规模向量整体趋势 | 余弦相似度 | `--compare cosine` |
| 整型/索引/掩码/纯搬移/纯比较 | 二进制精确 | `--compare binary` |
| hifloat8 量化 | 重量化 | `--compare requant`（自动） |
| int4/int8 量化输出 | 量化 | Spec.tolerance 声明 `quant` |
| 多硬件精度对齐 | 三方交叉校验 | `--compare cross_check`（需 XPU） |
