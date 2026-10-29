# 性能测试

## 使用须知

`--warmup`、`--run`、`--task-prof` 控制预热与采集行为，区分稳态与冷启动；各通路采集机制不同。

文档导览：

1. **[核心参数](#一核心参数)**
2. **[性能模式](#二性能模式)**
3. **[通路机制](#三通路机制)**
4. **[结果字段](#四结果字段)**

## 一、核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--warmup` | `true` | 采集前是否预热 |
| `--run` | `3`（仿真模式 `1`） | 采集迭代次数 |
| `--task-prof` | `true` | 任务级性能采集开关；`false` 时不采集性能，性能列无有效数据 |

## 二、性能模式

| 模式 | 参数 | 语义 |
|------|------|------|
| 稳态性能（默认） | `--warmup true --run 3` | 预热后跑 3 次（kernel/aclnn/geir 首轮不进统计、e2e 全部计入） |
| 冷启动性能 | `--warmup false --run 1` | 不预热，只跑 1 次 |

```shell
python3 -m ttk e2e -i cases.csv -o warm.csv                        # 稳态性能（默认）
python3 -m ttk e2e -i cases.csv --warmup false --run 1 -o cold.csv # 冷启动性能
```

> [!TIP]
> 两次运行加 `--seed 42` 固定输入，冷热对比只差预热。

## 三、通路机制

| 通路 | warmup | 采集方式 | 统计 |
|------|--------|---------|------|
| e2e（torch） | 预热被测算子 | torch_npu profiler | 平均 |
| e2e（tf） | 预热被测算子 | msprof | 平均 |
| e2e（onnx） | om + HBM 一次性装载 | 墙上时钟 | 平均 |
| kernel | 先跑 1 次辅助 kernel 预热 | msprof | 中位数（首轮不计） |
| aclnn | 同 kernel | msprof | 平均（首轮不计） |
| geir | 首轮即预热 | msprof | 中位数（首轮不计） |

> [!IMPORTANT]
> kernel / aclnn：即使 `--warmup false --run 3`，第一次执行也不计入统计；要采集首次执行需 `--run 1`。

## 四、结果字段

| 通路 | 性能列 |
|------|--------|
| kernel | `dyn_perf_us`、`cst_perf_us`、`bin_perf_us` |
| geir | `cst_perf_us`、`dyn_perf_us`（二进制模式为 `cst_bin_perf_us`、`dyn_bin_perf_us`） |
| aclnn | `api_perf_us`、`op_perf_us` |
| e2e | eager 列：`eager_device_perf_us`、`eager_cpu_perf_us`、`eager_kernel_count`、`eager_kernel_details`；图模式另有 `graph_cst_*`、`graph_dyn_*`、`graph_aclgraph_*` 对应列族。onnx 的 `eager_device_perf_us` 为墙上时钟计时，kernel 明细列为空；`--eager -c` 双开时 EAGER 与 GRAPH CST 两列同值 |

`kernel_details` 为 JSON 数组，含每个 kernel 的 `name`/`total`/`avg`/`max`/`min`/`calls`。
