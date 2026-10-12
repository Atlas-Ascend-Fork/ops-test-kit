# 精度调试详解

## 比对方法

| 方法 | 参数 | 原理 | 适用场景 |
|------|------|------|---------|
| 混合容差（生态算子开源精度标准） | `--compare mixed` | `\|a-g\| <= atol + rtol*\|g\|` 逐元素 + 通过率 ≥ 0.99 + 绝对误差硬上限 | 浮点运算常规测试（默认） |
| 统计相对误差（社区标准） | `--compare stat_rel_err` | 按 dtype 统计相对误差，社区标准阈值 | 统计误差均值/最大值 |
| 数值近似 | `--compare close` | `np.allclose(a, b, rtol, atol)` | 逐点 isclose 比对 |
| 余弦相似度 | `--compare cosine` | 向量余弦距离 | 大规模向量整体趋势验证 |
| 二进制精确 | `--compare binary` | 逐字节比对 | 整型运算、需要完全一致的结果 |
| 重量化 | `--compare requant` | 重量化后比对 | 量化数据类型（hifloat8 自动启用；float8 e4m3fn/e5m2 默认混合容差，可显式指定 requant；float8 e8m0 默认二进制比对） |
| 三方交叉校验 | `--compare cross_check` | output/golden/third_party 误差比值 | 有 third_party 参考实现时（fp16/bf16/fp32） |

> 默认未设 `--compare` 时，按 `Spec.tolerance` 逐输出路由（需 `--plugin`），否则 `mixed`。

## 容差参数

容差来源优先级：`--compare`（CLI）> `Spec.tolerance`（TestSpec，需 `--plugin`）> 方法默认阈值。CSV `precision_tolerances`/`absolute_precision` 为 legacy 字段：`precision_tolerances` 由 `close`/`cosine` 读取，`absolute_precision` 仅 `close` 读取；默认 `mixed` 不读 CSV 字段。

### Spec.tolerance（首选）

在 TestSpec 中按 dtype 声明精度标准（含 `standard` token + 阈值），逐输出路由：

```python
# TestSpec：float32 走 mix_tolerance（生态算子开源精度标准，默认）
tolerance = {"float32": {"standard": "mix_tolerance", "rtol": 0.002}}
```

### CSV 字段

| 字段 | 默认 | 说明 |
|------|------|------|
| `precision_tolerances` | None | 每个输出的 `(rtol, ptol)` 对。如 `"((0.001, 0.001),)"`。legacy 字段，仅 `--compare close`/`cosine` 读取 |
| `absolute_precision` | 1e-8 | 全局绝对精度容差。可以是单值或嵌套逐输出控制。legacy 字段，仅 `--compare close` 读取 |

### 容差设置示例

```csv
# 单输出：rtol=0.001, atol=0.001
precision_tolerances,"((0.001, 0.001),)"

# 多输出：每个输出独立设置
precision_tolerances,"((0.001, 0.001), (0.01, 0.01))"

# 全局绝对精度
absolute_precision,1e-5
```

## Dump 数据工作流

### 1. 收集数据

```shell
# 失败时自动 dump 全部数据
python3 -m ttk kernel -i cases.csv -t case_name --dump-on-fail

# 指定 dump 内容和格式
python3 -m ttk kernel -i cases.csv -t case_name \
  --dump in,out,golden --dump-format npy
```

### 2. 分析数据

```python
import numpy as np

# 加载 dump 数据：目录 <root>/dump/<case_name>/
# 文件名格式 {role}[_{segment}]_{idx}_{dtype}.npy（Kernel 输入/输出带 dyn/cst/bin 段）
# 先用 ls 确认实际文件名，例如：
input_data = np.load('dump/case_name/input_dyn_0_float32.npy')
output_data = np.load('dump/case_name/output_dyn_0_float32.npy')
golden_data = np.load('dump/case_name/golden_0_float32.npy')

# 计算差异
diff = np.abs(output_data - golden_data)
max_diff = np.max(diff)
mean_diff = np.mean(diff)
print(f"最大差异: {max_diff}, 平均差异: {mean_diff}")

# 检查 NaN/Inf
print(f"NaN 数量: {np.sum(np.isnan(output_data))}")
print(f"Inf 数量: {np.sum(np.isinf(output_data))}")
```

### 3. dump 格式选项

| 格式 | 参数 | 说明 |
|------|------|------|
| 二进制 | `--dump-format bin` | 原始二进制（默认） |
| NumPy | `--dump-format npy` | npy 格式，方便 Python 加载 |
| PyTorch | `--dump-format pt` | pt 格式 |
| 打印 | `--dump-format print` | 直接打印到日志 |

## 常见精度问题

| 现象 | 原因 | 修复 |
|------|------|------|
| 输出全为 NaN | 输入范围导致溢出 | 缩小 `input_data_ranges` |
| 输出全为 0 | 算子实现缺陷 | 检查 Golden 函数是否正确 |
| 大误差集中在边界值 | 数值溢出/下溢 | 调整输入范围或容差 |
| 余弦相似度 > 0.99 但 close 失败 | 个别极值偏差 | 放大 `absolute_precision` 或用 `--compare cosine` |
| 二进制失败但数值接近 | 浮点舍入差异 | 用 `--compare close` + 合理容差 |
| 特定 shape 才失败 | Tiling 分块边界问题 | Dump 数据分析边界值行为 |

## 调试最佳实践

1. **先跑内置示例**：`python3 -m ttk kernel -i examples/case_store/kernel/cases.csv -t add_fp16_broadcast` 确认环境正常
2. **固定种子**：`--seed 42` 确保可复现
3. **逐步放大**：先用小 shape 验证逻辑，再放大测试性能
4. **锁定单用例**：`-t case_name` 隔离问题（日志在 `log/<mode>/<用例名>/ttk.log`）
5. **对比 Golden**：先确认 Golden 函数正确，再比对算子输出
