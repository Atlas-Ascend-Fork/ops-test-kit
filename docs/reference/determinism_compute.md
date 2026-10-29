# 确定性计算与 batch 一致性

## 使用须知

`--deterministic-level`（`--dl`）检查"同一输入是否总是得到同一输出"：

- 同一用例重复执行、跨卡、跨进程时，输出应逐位一致——用来发现算子对执行顺序、并发或随机状态的隐式依赖
- level 3 且声明 batch 三字段时，进一步比对 batch 内各样本切片（同 seed 的样本应一致，如整批跑与分批跑结果一致）

## 文档导览

1. **[等级用法](#一等级用法)**
2. **[batch 字段](#二batch字段)**
3. **[比对方式](#三比对方式)**

## 一、等级用法

| 等级 | 含义 |
|------|------|
| `0` | 不启用（默认） |
| `1` | 同一用例重复执行结果一致 |
| `2` | 强一致（仅 aclnn / e2e 生效） |
| `3` | 确定性执行；声明 batch 三字段后进入 batch 一致性比对 |

也可在 CSV 的 `attributes` 里写 `'batch_deterministic_level': 3`（仅当命令行 `--dl` 为 0 时生效，非零命令行优先）。该属性只设等级，不视为 batch 三字段。

## 二、batch 字段

`batch_axis` / `batch_slice_info` / `batch_seed` 三个字段齐全时构成 batch 声明：框架对每个声明切片用独立 seed 生成，同 `(seed, 区段, dtype, 范围, 分布)` 的样本跨用例逐位一致。

三字段不全时 level 3 照常执行，但不做 batch 比对（执行前会提示缺字段）。

## 三、比对方式

### 1. 第一阶段（框架内）

按声明切片对第 0 个输出逐样本计算哈希，跨用例按 (api + seed + 切片长度) 分组比较是否一致；结果 CSV 含 `batch_consistency_id` 列。

### 2. 第二阶段（dump 驱动，可选）

不同 seed 的整批与分批比对由 `scripts/compare_batch_consistency.py` 完成：先以 `--dump out --dump-format bin` 执行（设 `NPU_DUMP_PATH`），再用结果 CSV 跑脚本比对 dump。

> [!IMPORTANT]
> 脚本要求 dump 文件平铺命名（`NPU_DUMP_PATH/<用例名>_output_0.bin`）；`--dump` 生成的是目录布局，使用前需按脚本约定整理。
