# 性能测试

TTK 支持采集算子在 NPU 上的执行性能。通过 `--warmup`、`--run`、`--task-prof` 等参数控制预热与采集行为，区分稳态性能与冷启动性能。

## 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--warmup` | `true` | 采集前是否预热 |
| `--run` | `3`（仿真模式 `1`） | 采集迭代次数 |
| `--task-prof` | `true` | 任务级性能采集开关；`false` 时关闭 msprof/torch profiler 采集，仅保留功能与精度比对，性能列显示占位值（kernel：`OK,OK,OK`；geir：`CST_OFF`/`DYN_OFF`；aclnn：空 + `perf_status=UNKNOWN`；e2e：`----`） |

## 两种性能模式

| 模式 | 参数 | 语义 |
|------|------|------|
| 稳态性能（默认） | `--warmup true --run 3` | 预热后采集 3 次（统计口径随通路：剔除首轮取中位数/均值等，见下文），反映算子稳态开销 |
| 冷启动性能 | `--warmup false --run 1` | 不预热，只跑 1 次，反映首次执行的全部开销 |

```shell
# 稳态性能（默认）
python3 -m ttk e2e -i cases.csv -o warm.csv

# 冷启动性能
python3 -m ttk e2e -i cases.csv --warmup false --run 1 -o cold.csv
```

> 两次运行加 `--seed 42` 保证输入数据一致，使冷热对比只反映预热差异。

## 各通路机制

不同通路的 warmup 和采集机制不同，性能数据口径需分别理解：

### e2e（框架 API）

- **warmup**：把被测算子本身跑 `WARMUP_COUNT`（当前为 1）次，结果丢弃；预热框架 dispatch、autograd、NPU kernel 首次加载、workspace 分配、SMMU/TLB
- **采集**：`--run` 次迭代全部进入 torch profiler 采集窗口，取平均值
- **冷启动**：`--warmup false` 跳过预热，`--run 1` 只跑 1 次

### kernel（AscendC 算子内核）

- **warmup**：launch 空 helper kernel `warmup.o` 1 次（每个 core 仅 `tik_return`，无实际计算），预热 SMMU/BIU/TLB/cores
- **采集**：`start_step=0`，`--run` 次迭代全部进入采集窗口，统计时剔除首轮（aclprof 异步启动的首 launch 计时不准），对剩余迭代取 median；`--run=1` 时取该 1 次
- **冷启动**：`--warmup false` 跳过 helper kernel，首次迭代含硬件冷启动开销

### aclnn（aclnn C API）

- **warmup**：同 kernel，launch 空 helper kernel 1 次，仅预热硬件层
- **采集**：`--run > 1` 时 `start_step=1` 跳过首次，采后 N-1 次取平均。首次执行需分配 workspace 等 HBM 内存（`aclnnAddGetWorkspaceSize` 首次耗时远高于后续），属于一次性内存申请开销，不计入稳态性能；`--run 1` 时 `start_step=0` 采那 1 次
- **冷启动**：`--warmup false --run 1`，不预热且只跑 1 次，采到完整冷启动开销
- **注意**：`--warmup false --run 3` 仍会跳首次（`start_step=1`），因为 `run_time > 1`；若要采首次需用 `--run 1`

### geir（GE 图编译执行）

- **warmup**：无
- **采集**：非确定性模式（默认）只跑 1 次子进程，取 `median`；确定性模式（`--dl 1`）跑 `--run` 次做 MD5 校验
- **冷启动**：geir 每次都是全新子进程执行编译好的二进制，冷热启动等价，`--warmup` 开关无效

## 采集次数与统计方式

| 通路 | 采集工具 | 读取文件 | 读取列 | 统计方式 | 实际采集次数 |
|------|---------|---------|--------|---------|-------------|
| kernel | CANN msprof | `op_summary_*.csv` / `task_time_*.csv` | `Task Duration(us)` / `task_time(us)` | `median`（中位数，剔除首轮） | `--run` 次 |
| geir | CANN msprof（子进程） | `op_summary_*.csv` / `task_time_*.csv` | `Task Duration(us)` / `task_time(us)` | `median`（中位数） | `1` 次（默认）；`--run` 次（`--dl 1`） |
| aclnn | CANN msprof | `api_statistic_*.csv` / `op_statistic_*.csv` | `Avg(us)` / `Avg Time(us)` | msprof 内部按采集次数取平均 | `--run - 1` 次（`--run > 1`）；`1` 次（`--run = 1`） |
| e2e（torch） | torch_npu.profiler | `kernel_details.csv` | `Duration(us)` | 累加后 `/ run_count`（平均值） | `--run` 次 |
| e2e（tf） | CANN msprof | `op_statistic_*.csv` | `Avg Time(us)` | msprof 内部按采集次数取平均，求和所有 op | `--run` 次 |

> kernel/geir 读取 `op_summary_*`/`task_time_*` 时按 `Task Type`/`kernel_type` 过滤，仅保留 `KERNEL_AICORE`/`KERNEL_AIVEC` 等实际 NPU kernel 执行记录，排除 `MODEL_EXECUTE`/`NOTIFY_*` 等框架开销；aclnn 读取 `api_statistic_*`/`op_statistic_*` 聚合表（api 行按 `Level=acl` 过滤）。

## 结果字段

结果 CSV 中各通路性能相关列：

| 通路 | 性能列 |
|------|--------|
| kernel | `dyn_perf_us`、`cst_perf_us`、`bin_perf_us` |
| geir | `cst_perf_us`、`dyn_perf_us`（二进制模式为 `cst_bin_perf_us`、`dyn_bin_perf_us`） |
| aclnn | `api_perf_us`、`op_perf_us` |
| e2e | eager 列：`eager_device_perf_us`、`eager_cpu_perf_us`、`eager_kernel_count`、`eager_kernel_details`；图模式另有 `graph_cst_*`、`graph_dyn_*`、`graph_aclgraph_*` 对应列族（device_perf_us/cpu_perf_us/kernel_count/kernel_details） |

`kernel_details` 为 JSON 数组，含每个 kernel 的 `name`/`total`/`avg`/`max`/`min`/`calls` 字段，可用于分析单 kernel 耗时与调用次数。
