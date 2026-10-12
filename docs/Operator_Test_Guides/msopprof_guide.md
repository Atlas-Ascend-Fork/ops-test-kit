# msopprof适配方法指南


---

# 概述

[msopprof](https://gitcode.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_user_guide.md) 是 MindStudio 提供的算子级性能采集与分析工具，可采集算子执行过程中的硬件流水、带宽利用率等细粒度数据，并生成流水图（`visualize_data.bin`）供 MindStudio Insight 查看。

TTK 内部自带 Profiling 采集流程，与 msopprof 的数据采集存在冲突。因此使用 msopprof 包裹 TTK 执行时，必须通过参数关闭 TTK 内部的 Profiling 采集与预热：

```
--task-prof=false --warmup=false
```

# 环境准备

## 基本环境配置

- 建议Python 3.8+
- 完成CANN包安装

## 安装 MindStudio 算子工具

下载与目标平台架构匹配的 MindStudio RUN包（建议使用较新的 RUN 包，旧版包可能存在适配问题）。将下方命令中的 `${arch}` 替换为下载包的架构标识（如 `x86` 或 `aarch64`）：

安装命令：

```shell
bash ascend-mindstudio-operator-tools_26.2.0_${arch}.run --install-path=<install_path> --run
```

设置环境变量：

```shell
source Ascend/cann/set_env.sh
```

> 注：MindStudio算子工具还需要 firmware 和 driver 包的适配，请关注这两个包是否安装正确。

# 调用 msopprof 命令

## 基本用法

使用 `msopprof` 包裹 `ttk kernel` 命令，并务必添加 `--task-prof=false --warmup=false` 关闭 TTK 内部 Profiling：

```shell
msopprof python3 -m ttk kernel -i examples/case_store/kernel/cases.csv -t add_fp16_broadcast --task-prof=false --warmup=false
```

## 结合 TTK 筛选参数

msopprof 包裹的 TTK 命令可以使用 TTK 自带的用例筛选与编译参数：

```shell
# 按索引范围运行
msopprof python3 -m ttk kernel -i cases.csv --task-prof=false --warmup=false --ti=0-9

# 按算子名筛选
msopprof python3 -m ttk kernel -i cases.csv --task-prof=false --warmup=false --op add,mat_mul_v3

# 按优先级筛选
msopprof python3 -m ttk kernel -i cases.csv --task-prof=false --warmup=false --priority=1-3

# 随机选取10个用例
msopprof python3 -m ttk kernel -i cases.csv --task-prof=false --warmup=false --tc=10

# 运行Add算子（动态 shape 编译）
msopprof python3 -m ttk kernel -i examples/case_store/kernel/cases.csv -t add_fp16_broadcast -d --task-prof=false --warmup=false

# 运行Add算子（二进制）
msopprof python3 -m ttk kernel -i examples/case_store/kernel/cases.csv -t add_fp16_broadcast --binary --task-prof=false --warmup=false
```

## 附加 msopprof 参数

可在 `msopprof` 与 `python3` 之间添加 msopprof 自身参数，例如采集时序数据：

```shell
# 时序命令
msopprof --aic-metrics=TimelineDetail python3 -m ttk kernel -i cases.csv --task-prof=false --warmup=false
```

更多 msopprof 参数详见：[msopprof 用户指南](https://gitcode.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_user_guide.md)

# 生成数据与分析

执行后会在输出目录生成性能数据文件，其中 `visualize_data.bin` 为流水图文件，L2Cache.csv ， MemoryL0.csv等为性能数据。

## 查看流水图

- **MindStudio Insight 工具**：使用 MindStudio Insight 工具查看流水图。建议 MindStudio Insight 的版本与 MindStudio 算子工具保持一致。参考：MindStudio Insight 工具。

## CSV 文件参数解读

生成的 CSV 文件中关键性能指标说明：

| 指标 | 说明 |
|------|------|
| `aic_main_mem_read_bw(GB/s)` | 从 GM 往外搬运的带宽，不只是 GM 到 L1 这部分带宽 |
| `GM_to_L1_datas(KB)` | GM 到 L1 的搬运数据量 |
| `GM_to_L1_bw_usage_rate(%)` | GM 到 L1 的带宽利用率 |
| `aic_L1_read_bw(GB/s)` | L1 到 L0A 和 L0B 的带宽（目前暂无途径分别采集两个带宽的数据） |
| `aiv time` | Block 级子核时间 |
| `task duration` | 算子 task 粒度的实际耗时，包含硬件调度、不包含软件调度 |

关于 `aiv time` 与 `task duration` 的关系：

- `task duration` 一般大于 `aiv time`，因为 `task duration` 包含硬件调度时间，而 `aiv time` 是纯核执行时间。
- 从数据来源看，`task duration` 由 stars 采集，`aiv time` 由 ffts 采集。
