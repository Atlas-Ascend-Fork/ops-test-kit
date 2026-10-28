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
| 稳态性能（默认） | `--warmup true --run 3` | 预热后跑 3 次（kernel/aclnn/geir 首轮均不进统计、e2e 全部计入，见下文），反映算子稳态开销 |
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

### e2e onnx 域（atc→om→aclmdl）

- **warmup**：warmup 轮完成一次性 setup（om 编译命中缓存后的装载、HBM 分配、H2D），轮尾重置计时窗
- **采集**：**墙钟计时**（aclmdl 执行不经 torch dispatcher，无 torch_npu profiler 采集窗）——计时轮只走 `aclmdlExecute` + D2H，对 `--run` 轮取平均；**kernel 明细为空**（已知限制，算子级明细请用 kernel/aclnn/geir 通路）
- **冷启动**：`--warmup false` 时计时窗含 om 装载与 H2D 的一次性开销

详见 [ONNX算子测试指南](./Operator_Test_Guides/ONNX_Test_Guide.md#性能测试)。

### kernel（AscendC 算子内核）

- **warmup**：launch 空 helper kernel `warmup.o` 1 次（每个 core 仅 `tik_return`，无实际计算），预热 SMMU/BIU/TLB/cores
- **采集**：`--run > 1` 时 `start_step=1` 首轮不进采集窗口（aclprof 异步启动的首 launch 计时不准 + kernel 首次装载开销，与 aclnn 同口径），对窗口内迭代取 median；`--run = 1` 时 `start_step=0` 采那 1 次
- **冷启动**：`--warmup false` 跳过 helper kernel，首次迭代含硬件冷启动开销
- **注意**：`--warmup false --run 3` 仍会跳首次（`start_step=1`），因为 `run_time > 1`；若要采首次需用 `--run 1`

### aclnn（aclnn C API）

- **warmup**：同 kernel，launch 空 helper kernel 1 次，仅预热硬件层
- **采集**：`--run > 1` 时 `start_step=1` 跳过首次，采后 N-1 次取平均。首次执行需分配 workspace 等 HBM 内存（`aclnnAddGetWorkspaceSize` 首次耗时远高于后续），属于一次性内存申请开销，不计入稳态性能；`--run 1` 时 `start_step=0` 采那 1 次
- **冷启动**：`--warmup false --run 1`，不预热且只跑 1 次，采到完整冷启动开销
- **注意**：`--warmup false --run 3` 时预热轮仍不进窗（`start_step=1`）；若要采首轮需用 `--run 1`

### geir（GE 图编译执行）

- **warmup**：无（`--warmup` 开关不改变轮数；预热轮即首轮，见下）
- **采集**：执行 `--run` 轮（与 kernel/aclnn 同口径），结束后统一解析本次新增的 PROF 目录（按目录名内嵌时间戳升序，与执行轮次一一对应）；时间戳最早的**首轮**（图编译/装载冷启动）不进窗口，其余 `--run - 1` 轮 Task Duration 跨轮取 `median`；`--run 1` 时 `start_step` 语义等价——唯一一轮计入（冷启动口径）；`--dl 1` 时全部轮同时做 MD5 确定性校验
- **冷启动**：serve 常驻子进程整批一次 GEInitialize；每用例每模式时间戳最早的 PROF 目录即首轮冷启动原始数据（TTK 未对其执行 msprof 导出，需要时用 msprof 自行导出 `task_time`/`op_summary` 后读取）

## 采集次数与统计方式

| 通路 | 采集工具 | 读取文件 | 读取列 | 统计方式 | 实际采集次数 |
|------|---------|---------|--------|---------|-------------|
| kernel | CANN msprof | `op_summary_*.csv` / `task_time_*.csv` | `Task Duration(us)` / `task_time(us)` | `median`（中位数） | `--run - 1` 次（`--run > 1`，首轮不进窗）；`1` 次（`--run = 1`） |
| geir | CANN msprof（serve 子进程） | `op_summary_*.csv` / `task_time_*.csv` | `Task Duration(us)` / `task_time(us)` | `median`（轮内任务级 + 跨轮） | `--run - 1` 次（`--run > 1`，首轮不进窗）；`1` 次（`--run = 1`） |
| aclnn | CANN msprof | `api_statistic_*.csv` / `op_statistic_*.csv` | `Avg(us)` / `Avg Time(us)` | msprof 内部按采集次数取平均 | `--run - 1` 次（`--run > 1`）；`1` 次（`--run = 1`） |
| e2e（torch） | torch_npu.profiler | `kernel_details.csv` | `Duration(us)` | 累加后 `/ run_count`（平均值） | `--run` 次 |
| e2e（tf） | CANN msprof | `op_statistic_*.csv` | `Avg Time(us)` | msprof 内部按采集次数取平均，求和所有 op | `--run` 次 |
| e2e（onnx） | 墙钟（SteppedWallClockProfiler） | — | — | warmup 后计时窗 `/ run_count`（平均值） | `--run` 次 |

> kernel/geir 读取 `op_summary_*`/`task_time_*` 时按 `Task Type`/`kernel_type` 过滤，仅保留 `KERNEL_AICORE`/`KERNEL_AIVEC` 等实际 NPU kernel 执行记录，排除 `MODEL_EXECUTE`/`NOTIFY_*` 等框架开销；aclnn 读取 `api_statistic_*`/`op_statistic_*` 聚合表（api 行按 `Level=acl` 过滤）。

## 结果字段

结果 CSV 中各通路性能相关列：

| 通路 | 性能列 |
|------|--------|
| kernel | `dyn_perf_us`、`cst_perf_us`、`bin_perf_us` |
| geir | `cst_perf_us`、`dyn_perf_us`（二进制模式为 `cst_bin_perf_us`、`dyn_bin_perf_us`） |
| aclnn | `api_perf_us`、`op_perf_us` |
| e2e | eager 列：`eager_device_perf_us`、`eager_cpu_perf_us`、`eager_kernel_count`、`eager_kernel_details`；图模式另有 `graph_cst_*`、`graph_dyn_*`、`graph_aclgraph_*` 对应列族（device_perf_us/cpu_perf_us/kernel_count/kernel_details）。onnx 域 `eager_device_perf_us` 为墙钟、kernel 明细列为空；`--eager -c` 双开时 EAGER 与 GRAPH CST 两列同值（单次执行镜像） |

`kernel_details` 为 JSON 数组，含每个 kernel 的 `name`/`total`/`avg`/`max`/`min`/`calls` 字段，可用于分析单 kernel 耗时与调用次数。
