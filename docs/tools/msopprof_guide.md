# msopprof 适配方法

## 使用须知

[msopprof](https://gitcode.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_user_guide.md) 是 MindStudio 的算子级性能采集工具，输出硬件流水、带宽利用率等数据与流水图（`visualize_data.bin`，供 MindStudio Insight 查看）。包裹 `ttk kernel` 使用，需关闭 TTK 内部 Profiling 与预热。

文档导览：

1. **[环境准备](#一环境准备)**
2. **[基本用法](#二基本用法)**
3. **[数据分析](#三数据分析)**

## 一、环境准备

- Python 3.8+
- 完成 CANN 包安装
- 下载与目标平台架构匹配的 MindStudio RUN 包并安装（`${arch}` 为 `x86` 或 `aarch64`）：

```shell
bash ascend-mindstudio-operator-tools_26.2.0_${arch}.run --install-path=<install_path> --run
source Ascend/cann/set_env.sh
```

> [!NOTE]
> 工具还需 firmware 和 driver 包适配，请确认这两个包安装正确。

## 二、基本用法

> [!IMPORTANT]
> TTK 内部自带 Profiling，与 msopprof 采集冲突，包裹执行时必须关闭：`--task-prof=false --warmup=false`。

### 1. 包裹 TTK 命令

```shell
msopprof python3 -m ttk kernel -i examples/case_store/kernel/cases.csv -t add_fp16_broadcast --task-prof=false --warmup=false
```

TTK 的筛选与编译参数照常可用：

```shell
msopprof python3 -m ttk kernel -i cases.csv --task-prof=false --warmup=false --ti=0-9
msopprof python3 -m ttk kernel -i cases.csv --task-prof=false --warmup=false --op add,mat_mul_v3
msopprof python3 -m ttk kernel -i examples/case_store/kernel/cases.csv -t add_fp16_broadcast -d --task-prof=false --warmup=false
```

### 2. 附加 msopprof 参数

msopprof 自身参数放在 `msopprof` 与 `python3` 之间：

```shell
msopprof --aic-metrics=TimelineDetail python3 -m ttk kernel -i cases.csv --task-prof=false --warmup=false
```

更多参数详见 [msopprof 用户指南](https://gitcode.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_user_guide.md)。

## 三、数据分析

执行后在输出目录生成性能数据，`visualize_data.bin` 为流水图，另有 L2Cache.csv、MemoryL0.csv 等性能数据。

### 1. 查看流水图

用 MindStudio Insight 查看（版本宜与算子工具一致）。

### 2. 关键指标

| 指标 | 说明 |
|------|------|
| `aic_main_mem_read_bw(GB/s)` | 从 GM 往外搬运的带宽 |
| `GM_to_L1_datas(KB)` | GM 到 L1 的搬运数据量 |
| `GM_to_L1_bw_usage_rate(%)` | GM 到 L1 的带宽利用率 |
| `aic_L1_read_bw(GB/s)` | L1 到 L0A/L0B 的带宽 |
| `aiv time` | Block 级子核时间 |
| `task duration` | 算子 task 粒度实际耗时（含硬件调度） |

`task duration` 一般大于 `aiv time`：前者含硬件调度时间，`aiv time` 是纯核执行时间。
