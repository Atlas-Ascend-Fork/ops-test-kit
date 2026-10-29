# 任务执行

## 使用须知

TTK 通过 `python3 -m ttk <子命令>` 批量执行测试（需在仓库目录下）。本文列出各模式通用的命令行参数，各通路专属参数见对应指南。

文档导览：

1. **[命令总览](#一命令总览)**
2. **[设备与预览](#二设备与预览)**
3. **[通用参数](#三通用参数)**
4. **[专属参数](#四专属参数)**
5. **[高阶场景](#五高阶场景)**

## 一、命令总览

```
python3 -m ttk {kernel,aclnn,e2e,geir,info,list,help} [选项]
```

| 子命令 | 用途 |
|--------|------|
| `kernel` | AscendC 算子内核编译 + NPU 执行 + 精度比对 |
| `geir` | GE 图编译 + 执行 + 精度比对 |
| `aclnn` | aclnn\* C API 调用 + 精度比对 |
| `e2e` | 框架 API 端到端测试（torch / tensorflow / onnx） |
| `info` | 查询本机 Ascend NPU 设备信息 |
| `list` | 预览用例文件中的测试用例列表 |
| `help` | 命令帮助与参数说明（`ttk help [COMMAND] [PARAM]`） |

```shell
python3 -m ttk -v            # 查看版本
python3 -m ttk kernel --help # 查看帮助（aclnn/e2e/geir 同理）
```

## 二、设备与预览

```shell
python3 -m ttk info                                  # 查看设备信息
python3 -m ttk list -i examples/case_store/kernel/cases.csv   # 预览用例列表
python3 -m ttk list -i cases.csv --op add            # 按算子名过滤
python3 -m ttk list -i cases.xlsx --sheet Sheet2     # 预览 Excel（指定工作表）
```

> [!NOTE]
> `ttk list` 仅列出用例名（支持 `--op`/`--sheet` 过滤），只要求 `testcase_name` 列存在且非空，不做其他字段校验；完整校验用对应模式的 `--validate`。

## 三、通用参数

### 1. 用例筛选

| 参数 | 缩写 | 默认值 | 说明 | 示例 |
|------|------|--------|------|------|
| `--input` | `-i` | 无 | 用例文件路径，支持 csv / xlsx / xlsm / zip（必填） | `-i add.csv` |
| `--sheet` | | 首个工作表 | Excel 工作表名，仅 `.xlsx`/`.xlsm` 生效 | `--sheet T2` |
| `--testcase` | `-t` | 无 | 按用例名筛选（逗号分隔） | `-t add_01,add_02` |
| `--testcase-index` | `--ti` | 无 | 按用例索引筛选（从 0 起） | `--ti=0-9` |
| `--testcase-count` | `--tc` | 无 | 随机选取 N 个用例 | `--tc=10` |
| `--operator` | `--op` | 无 | 按算子名筛选 | `--op add,mat_mul_v3` |
| `--exclude-operator` | `--no-op` | 无 | 排除算子名 | `--no-op concat_d` |
| `--priority` | | 无 | 按优先级筛选（单值/闭区间/开放上界） | `--priority=1-3` |
| `--provider` | | 无 | 三方 provider 过滤（如 torch/tf） | `--provider torch` |
| `--rerun` | | 无 | 重跑失败项（仅 Kernel/GEIR 支持） | `--rerun=precision_status` |
| `--config` | | 无 | ttk 配置 YAML 路径 | `--config ttk.conf.yaml` |

### 2. 设备与并行

| 参数 | 缩写 | 默认值 | 说明 | 示例 |
|------|------|--------|------|------|
| `--device` | `--dev` | 全部 | 使用设备数量 | `--dev 2` |
| `--devices` | | 无 | 设备 ID 表达式：`3` / `0,1` / `0-7`，`^` 剔除；可与 `--dev N` 组合取交集 | `--devices 0-7,^3` |
| `--process-count` | `--pc` | 自动 | 每张卡进程数（自动 = CPU 核数×80%÷卡数，限 1~4；仿真默认 1） | `--pc 4` |
| `--proc-no-reuse` | | 关闭 | 每个用例创建新进程 | `--proc-no-reuse` |
| `--no-memory-check` | | 关闭 | 跳过主机内存检查 | `--no-memory-check` |
| `--platform` | `--plat` | 自动检测 | SoC 版本 | `--plat Ascend910B3` |
| `--proc-timeout` | | 3600 | 单用例超时（秒） | `--proc-timeout 120` |
| `--limit` | `-l` | 30 | 单用例 HBM 内存上限（GB），超过判无效并跳过（kernel 按输入+输出计，aclnn 按输入计） | `-l 16` |

### 3. 执行控制

| 参数 | 缩写 | 默认值 | 说明 | 示例 |
|------|------|--------|------|------|
| `--run` | | 3（仿真 1） | 执行次数 | `--run=5` |
| `--task-prof` | | 开启 | 任务级 Profiling 开关 | `--task-prof=false` |
| `--warmup` | | 开启 | Profiling 前预热 | `--warmup=false` |
| `--npu-timeout` | | 无限制 | NPU 执行超时（ms），仅 kernel 通路 | `--npu-timeout 60000` |
| `--no-prof` | | 关闭 | 只生成输入和 Golden，不执行目标 API | `--no-prof` |
| `--validate` | | 关闭 | 仅校验 CSV 用例格式，不执行编译、输入生成、Golden 等流程 | `--validate` |

### 4. 精度控制

| 参数 | 缩写 | 默认值 | 说明 | 示例 |
|------|------|--------|------|------|
| `--compare` | | 未指定 | 精度对比方法：`close`/`stat_rel_err`/`mixed`/`cosine`/`binary`/`requant`/`cross_check`。未指定时按 Spec.tolerance 路由（需 `--plugin`）；无 Spec 声明时按输出 dtype 路由 | `--compare cosine` |
| `--input-dist` | | `uniform` | 输入数据分布：`uniform`/`normal` | `--input-dist normal` |
| `--seed` | | 随机 | 随机种子：固定后同一 CSV 连跑两次输入/golden 逐位一致，可用于改动前后 A/B 对比 | `--seed 42` |
| `--golden-mode` | | `Enable` | Golden 生成模式：`Enable`/`Disable`/`Promote`（CPU 高精度）/`XPU`（三方输出作 golden）/`NPU`（golden 上 NPU 执行）。`Disable` 时不做精度比对，结论为 `SKIP`。onnx：`XPU` 启动期拒绝，`NPU` 退化同 `Enable` | `--golden-mode NPU` |
| `--deterministic-level` | `--dl` | `0` | 确定性等级：`0` 关闭 / `1` 确定性计算 / `2` 强一致 / `3` 确定性执行 | `--dl=1` |

> [!NOTE]
> 精度比对方法的公式、容差及选择建议详见[精度比对方法](./reference/precision_comparison.md)。确定性计算详见[确定性计算与批一致性](./reference/determinism_compute.md)。

### 5. 调试与诊断

| 参数 | 缩写 | 默认值 | 说明 | 示例 |
|------|------|--------|------|------|
| `--dump` | | 关闭 | Dump 数据：`full`/`in`/`out`/`golden`/`xpu`，可逗号组合 | `--dump in,out,xpu` |
| `--dump-format` | | `bin` | Dump 格式：`bin`/`npy`/`pt`/`print` | `--dump-format npy` |
| `--dump-on-fail` | | 关闭 | 精度失败时自动补 dump 输入/输出/Golden | `--dump-on-fail` |
| `--manual-data-dirs` | | 无 | 离线数据目录（prepare 输出或 replay 搜索目录） | `--manual-data-dirs /data/op` |
| `--plugin` | | 无 | 外部插件路径 | `--plugin /path/to/plugin.py` |
| `--xpu-perf` | | 关闭 | 采集三方 XPU 性能（需远端 XPU 配置） | `--xpu-perf` |

### 6. 结果输出

| 参数 | 缩写 | 默认值 | 说明 | 示例 |
|------|------|--------|------|------|
| `--output` | `-o` | 无 | 输出结果 CSV 路径（覆盖已有文件），与 `-a` 互斥 | `-o results.csv` |
| `--append` | `-a` | 无 | 追加结果到已有 CSV（表头不匹配则覆盖），与 `-o` 互斥 | `-a results.csv` |
| `--po` / `--progress-output` | | 无 | 进度输出路径（`Key:Value` 纯文本） | `--po /tmp/progress.txt` |
| `--print` | | 开启 | 是否打印跑批过程中的周期性摘要 | `--print=false` |

> [!NOTE]
> 每个用例的独立日志目录是默认行为（`{root}/log/{mode}/{testcase_name}/ttk.log`），无需参数。结果文件名对空格/不安全字符做 `%XX` 转义（如 `A B` → `cases_A%20B_result.csv`），需自定义路径用 `-o`。

## 四、专属参数

- **Kernel 编译控制**（`-d`/`-c`/`-b`/`--co`/`--compile-opts` 等）：[Kernel 测试指南](./operator_test_guides/kernel_test_guide.md)
- **E2E 后端与编译**（`--backend cpu`、`-c`/`-d`/`--aclgraph` 等）：[E2E 测试指南](./operator_test_guides/e2e_test_guide.md)
- **GEIR 编译模式**（`-c`/`-d`/`-b`）：[GEIR 测试指南](./operator_test_guides/geir_test_guide.md)
- ACLNN 复用通用参数，无专属选项

## 五、高阶场景

### 1. 执行特性

| 文档 | 核心参数 |
|------|---------|
| [Dump 数据调试](./dump_debug_guide.md) | `--dump` + `--dump-format` + `--dump-on-fail` |
| [性能测试](./performance_testing_guide.md) | `--warmup` + `--run` |
| [确定性计算与批一致性](./reference/determinism_compute.md) | `--deterministic-level` / `--dl` |

### 2. 特殊场景使用技巧

| 文档 | 核心参数 |
|------|---------|
| [离线数据准备与导入](./tips/offline_data_prepare_guide.md) | `--no-prof --dump in,golden` + `--manual-data-dirs` |
| [三方交叉校验](./tips/cross_check_guide.md) | `--compare cross_check` / `--xpu-perf` + `--config` |
| [NPUSim 仿真测试](./npusim/npusim_usage_guide.md) | `--backend npusim` + `--sim-*` |
