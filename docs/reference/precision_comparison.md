# 精度比对方法

## 使用须知

用 `--compare` 选择比对方法，或由 TestSpec 的 `tolerance` 按 dtype 指定每个输出的比对方法。

常用缩写：

- `a` = NPU 输出，`g` = Golden（参考输出）
- `rtol` = 相对容差，`atol` = 绝对容差，`ptol` = 允许超差的元素比例
- `mere` = 平均相对误差，`mare` = 最大相对误差，`rmse` = 均方根误差
- `ULP` = 浮点相邻两数的最小间距，`LSB` = 最低有效位

| 方法 | 参数值 | 适用场景 | 默认容差 |
|------|--------|---------|---------|
| 混合容差 | `mixed`（默认） | 浮点算子常规测试 | 按 dtype 阈值表 |
| 统计相对误差 | `stat_rel_err` | 统计误差均值/最大值 | 按 dtype 阈值表 |
| 数值近似 | `close` | 逐点 isclose 比对 | 按 dtype 表，atol=1e-8 |
| 余弦相似度 | `cosine` | 大规模向量整体趋势 | 1-cos ≤ 0.01 |
| 二进制精确 | `binary` | 整型运算/精确结果 | 无容差 |
| 重量化 | `requant` | hifloat8 类型 | ptol=0.001 |
| 量化 | `quant` | 浮点输入 + int4/int8/uint8 量化输出 | ptol=0 |
| 三方交叉校验 | `cross_check` | 多硬件精度对齐 | 按 level 预设 |

按场景选择：

| 场景 | 方法 |
|------|------|
| 浮点算子 | `mixed`（默认）/ `stat_rel_err` / `close` |
| 整型、比较、搬移 | `binary` |
| hifloat8 | `requant`（自动） |
| 量化输出 | Spec 声明 `quant` |
| 多硬件对齐 | `cross_check` |

## 文档导览

1. **[混合容差（mixed）](#一混合容差mixed)**
2. **[统计相对误差（stat_rel_err）](#二统计相对误差stat_rel_err)**
3. **[数值近似（close）](#三数值近似close)**
4. **[余弦相似度（cosine）](#四余弦相似度cosine)**
5. **[二进制精确（binary）](#五二进制精确binary)**
6. **[重量化（requant）](#六重量化requant)**
7. **[量化（quant）](#七量化quant)**
8. **[三方交叉校验（cross_check）](#八三方交叉校验cross_check)**
9. **[方法选择与容差](#九方法选择与容差)**

## 一、混合容差（mixed）

默认方法（生态算子开源精度标准），按 atol + rtol 逐元素比对。

### 公式

```
元素通过：|a - g| <= atol + rtol * |g|
matched_ratio = 通过元素数 / 总元素数
passed = matched_ratio >= 0.99 AND max_abs_error <= max_abs_error_limit
```

- `atol`：绝对容差，保证小值（golden 接近 0）场景的合理误差，自动避免除零
- `rtol`：相对容差，保证大值场景的相对精度
- `max_abs_error_limit`：绝对误差上限。默认取「固定下限」与「32 × ULP(g_low)」的较大者，随 dtype 与值域自适应；Spec 显式配置后完全替代（固定值不随值域缩放）

### 默认容差

| dtype | rtol | atol | max_abs_error_limit |
|-------|------|------|---------------------|
| `float16` | 2^-9 (≈1.95e-3) | 2^-9 (≈1.95e-3) | 下限 1e-1，动态 32\*ULP |
| `bfloat16` | 2^-6 (≈1.56e-2) | 2^-6 (≈1.56e-2) | 下限 1e-0，动态 32\*ULP |
| `float32` | 2^-10 (≈9.77e-4) | 2^-16 (≈1.53e-5) | 下限 1e-2，动态 32\*ULP |
| `hifloat32` | 2^-9 (≈1.95e-3) | 2^-10 (≈9.77e-4) | 下限 1e-1，动态 32\*ULP |
| `float8_e4m3fn` | 2^-2 (0.25) | 2^-4 (0.0625) | 下限 1e-0，动态 32\*ULP |
| `float8_e5m2` | 2^-1 (0.5) | 2^-3 (0.125) | 下限 1e-1，动态 32\*ULP |
| 其他浮点（回退到 float32 档） | 2^-10 | 2^-16 | 下限 1e-2，动态 32\*ULP |

`required_matched_ratio` 全 dtype 默认值为 `0.99`（Spec 可用同名参数覆盖）。`hifloat8` 输出自动切换到[重量化](#六重量化requant)，不适用本表。

### INF/NaN 值处理

- 双方同位置均为 NaN → 视为一致
- 双方同位置均为 Inf 且同号 → 视为一致
- NaN/Inf 位置不一致 → 视为无界误差，必超上限，直接 FAIL

### 使用方式

```bash
python3 -m ttk kernel -i cases.csv --compare mixed

# Spec 声明（需 --plugin）
# tolerance = {"float32": {"standard": "mix_tolerance", "rtol": 0.002}}
```

## 二、统计相对误差（stat_rel_err）

按 dtype 统计相对误差的均值（mere）与最大值（mare），与阈值比较判定。

### 公式

```
rel_err = |a - g| / (|g| + 1e-7)
mere = mean(rel_err)       # 平均相对误差
mare = max(rel_err)        # 最大相对误差
passed = mere < threshold AND mare < 10 * threshold
```


- `mere` 超过 `threshold` → FAIL
- `mare` 超过 `10 * threshold` → FAIL（防止个别大误差被均值掩盖）

### 默认容差

| dtype | threshold | 说明 |
|-------|-----------|------|
| `float16` | 2^-10 (≈0.000977) | 半精度 |
| `bfloat16` | 2^-7 (≈0.007812) | BF16 精度较低，阈值放宽 |
| `float32` | 2^-13 (≈0.000122) | 单精度 |
| `float8_e4m3fn` | 2^-3 (≈0.125) | FP8 E4M3 |
| `float8_e5m2` | 2^-2 (≈0.25) | FP8 E5M2 |
| 其他 | 2^-13 | 回退到 float32 级 |

### INF/NaN 值处理

- NaN/Inf 位置不一致 → 直接 FAIL（不计算 mere/mare）
- 双方同位置均为 NaN → 视为一致
- 双方同位置均为 Inf 且同号 → 视为一致

### 使用方式

```bash
python3 -m ttk kernel -i cases.csv --compare stat_rel_err

# Spec 声明（需 --plugin）
# tolerance = {"float16": {"standard": "stat_rel_err"}}
```

## 三、数值近似（close）

等价 `np.isclose(..., equal_nan=True)` 的逐元素比对（fp16 先提升为 fp32），按 rtol/atol 两个容差判定。

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

```
precision = (golden_size - mismatch_count) / golden_size
passed = (1 - precision) <= ptol
```

### 自定义容差

```csv
precision_tolerances,"((0.001, 0.001),)"
absolute_precision,1e-8
```

- `precision_tolerances`：每个输出的 `(rtol, ptol)` 对
- `absolute_precision`：atol。单个浮点数应用于全部输出；tuple/list 时按输出索引逐输出取值，多输出算子各输出量纲不同时可分别设置：

```csv
absolute_precision,"(1e-8, 1e-5)"
```

> [!NOTE]
> 字段约定见[测试用例生成](../test_case_generation_guide.md)。

### 使用方式

```bash
python3 -m ttk kernel -i cases.csv --compare close

# Spec 声明（rtol/atol/ptol 可覆盖 dtype 默认；CSV 容差仍优先）
# tolerance = {"float16": {"standard": "isclose", "rtol": 0.005, "atol": 0.000025, "ptol": 0.005}}
```

## 四、余弦相似度（cosine）

计算输出与 Golden 的余弦相似度，衡量方向一致性。

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

### 使用方式

```bash
python3 -m ttk kernel -i cases.csv --compare cosine

# 仅 CLI 指定，Spec 不支持
```

## 五、二进制精确（binary）

逐 bit 比对 NPU 输出与 Golden，无容差概念。

```
output_bytes == golden_bytes → PASS
```

不一致时定位差异元素索引；同位置均为 NaN 视为相等。支持整型/布尔跨 dtype 比对（自动提升到公共类型）；浮点 vs 整型不兼容，直接拒绝。

### 适用场景

| 场景 | 说明 |
|------|------|
| 纯搬移算子 | 输出应与输入逐 bit 一致 |
| 纯比较算子 | 如 Equal、Less、Greater 等 |
| 整型运算 | int8/int32/uint 等精确结果 |
| 索引/掩码输出 | 非 0 即 1 的场景 |
| float4 / int4 | 打包格式逐 bit 比对 |


### 使用方式

```bash
python3 -m ttk kernel -i cases.csv --compare binary

# Spec 声明（需 --plugin）
# tolerance = {"float16": {"standard": "binary_equal"}}
```

## 六、重量化（requant）

面向 hifloat8，以 ULP 为判据。float8（e5m2/e4m3fn）默认走 mixed，也可用 `--compare requant` 显式选择。

### 公式

```
diff = |view_as_int8(a) - view_as_int8(g)|    # FP8/hifloat8 按 int8 视图比较 ULP 差
passed = (diff <= 1 的元素占比) >= (1 - ptol)
```

> [!NOTE]
> 非 ULP 视图 dtype（fp16/fp32/bf16 等）显式选择 requant 时，退化为绝对值判据：`|a - g| <= 1`（默认 ptol=0，即全部元素须满足）。

### 默认容差

| dtype | ptol |
|-------|------|
| `hifloat8`（自动切换） | 0.001（允许 0.1% 元素 ULP 差 > 1） |
| `float8_e5m2` / `float8_e4m3fn`（显式 `--compare requant`） | 0.001（允许 0.1% 元素 ULP 差 > 1） |

### 适用场景

- hifloat8 量化算子（自动切换）
- float8（e5m2/e4m3fn）显式选择 ULP 判据时

### 使用方式

```bash
# CLI
python3 -m ttk kernel -i cases.csv --compare requant
```

> [!NOTE]
> `hifloat8` 会自动切换到 `requant`；`float8_e5m2` / `float8_e4m3fn` 默认走 `mixed`（生态算子开源精度标准），也可通过 `--compare requant` 显式选择 ULP 判据。

## 七、量化（quant）

用于浮点输入 + int4/int8/uint8 量化输出的算子，判据为绝对误差 ≤ 1 LSB。

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
- 浮点输入 → int4/int8/uint8 输出
- 量化末步是阶跃函数，浮点末位 1 ULP 差异会导致结果跳相邻整数格，逐位相等没有意义

### 使用方式

只能通过 Spec.tolerance 声明（要求输出 int4/int8/uint8、输入含浮点）：

```python
__spec__ = {"per_token_quant": "PerTokenQuantTestSpec"}


class PerTokenQuantTestSpec:
    tolerance = {
        "int8": {"standard": "quant", "ptol": 0.001},
    }
```

## 八、三方交叉校验（cross_check）

比较 NPU 输出、Golden、第三方（XPU）输出的误差比值。用法见 [三方交叉校验与性能采集](../tips/cross_check_guide.md)。

### 公式

```
rel_npu   = |a - g| / (|g| + ε)          # NPU 相对误差
rel_party = |b - g| / (|g| + ε)          # XPU 相对误差

mare_ratio = max(rel_npu) / max(max(rel_party), small_value)   # 最大误差比
mere_ratio = mean(rel_npu) / max(mean(rel_party), small_value) # 平均误差比
rmse_ratio = rmse(a,g) / max(rmse(b,g), small_value)           # RMSE 比
```

> [!NOTE]
> 三个比值的分母都取 `small_value`（按 dtype 定的小值阈值）与实测值中的较大者：防除零，也避免第三方误差接近 0 时比值被放大。NaN 分子时比值为 inf；分子分母同号溢出或同为 0 时比值取 1。

大值域（`|g| ≥ small_value`）：三个比值均不超过 limit → PASS

小值域（`|g| < small_value`）：NPU 误差计数 / max(XPU 误差计数, 1) ≤ 2.0 → PASS

### INF/NaN 值处理

NPU 与 third_party 或 golden 任一方一致即通过（NaN/±Inf 同值亦算一致）；仅双方都不一致才 FAIL。第三方单方 NaN/Inf 不判失败。

### 等级预设

| Level | mare_ratio | mere_ratio | rmse_ratio | 适用 |
|-------|-----------|-----------|-----------|------|
| L0 | 10.0 | 2.0 | 2.0 | 宽松（初期对齐） |
| L1 | 5.0 | 1.5 | 1.5 | 默认 |
| L2 | 2.0 | 1.2 | 1.2 | 严格（接近对齐） |

### small_value

| dtype | small_value |
|-------|------------|
| `float16` | 2^-11 |
| `bfloat16` | 2^-8 |
| `float32` | 2^-14 |

> [!NOTE]
> `cross_check` 仅支持 `float16` / `bfloat16` / `float32`。

### 使用方式

```bash
python3 -m ttk kernel -i cases.csv --compare cross_check --config ttk.conf.yaml --plugin /path/to/assets.py

# Spec 声明
# tolerance = {"bfloat16": {"standard": "cross_check", "level": "L2"}}
```

> [!NOTE]
> 完整使用方法详见 [三方交叉校验与性能采集](../tips/cross_check_guide.md)。

## 九、方法选择与容差

### 1. 自动切换

部分 dtype 会自动切换比对方法，无需手动指定：

| 数据类型 | 自动切换到 | 原因 |
|---------|-----------|------|
| `hifloat8` | `requant` | 需 ULP 判据 |
| `float4` / `int4` | `binary` | 打包格式需逐 bit |
| `float8_e8m0` | `binary` | 无浮点语义 |
| `complex32` / `complex64` / `complex128` | `close`（isclose） | 复数需逐元素 close 判据 |
| 整型 / 布尔 | `binary` | 精确结果 |

### 2. 优先级

每个输出的比对方法按以下优先级决定：

```
1. CLI --compare 参数（最高优先级）
2. Spec.tolerance[dtype].standard（需 --plugin）
3. dtype 自动切换规则
4. mixed（默认）
```

> [!IMPORTANT]
> dtype 限制：整型/布尔、`float4`/`float8_e8m0` 强制 `binary`；complex 只接受 `binary`/`close`；`hifloat8` 只接受 `binary`/`requant`。`--compare` 仅对浮点类型无条件生效。

`quant` 只能经 Spec 声明；声明后 `--compare binary`/`requant` 可覆盖，其余浮点判据不覆盖。

### 3. 容差自定义

#### CSV 字段（旧版）

| 字段 | 作用 | 格式 | 读取方 |
|------|------|------|--------|
| `precision_tolerances` | 每输出的 `(rtol, ptol)` | `((0.001, 0.001),)` | `close`、`cosine`（仅 rtol）、`requant`/`quant`（仅 ptol，作为允许超差元素占比） |
| `absolute_precision` | atol；单值应用于全部输出，tuple/list 按输出索引逐输出配置 | `1e-8` / `(1e-8, 1e-5)` | 仅 `close` |

> [!NOTE]
> 默认 `mixed` 不读 CSV 字段，需在 TestSpec `tolerance` 中覆盖（需 `--plugin`）。

#### Spec.tolerance

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
