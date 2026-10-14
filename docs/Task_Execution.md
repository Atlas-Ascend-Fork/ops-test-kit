# 任务执行

# 命令行总览

```
python3 -m ttk {kernel,aclnn,e2e,geir,info,list,help} [选项]
```

| 子命令 | 用途 |
|--------|------|
| `kernel` | AscendC 算子内核编译 + NPU 执行 + 精度比对 |
| `geir` | GE 图编译 + 执行 + 精度比对 |
| `aclnn` | aclnn\* C API 调用 + 精度比对 |
| `e2e` | 框架 API 端到端测试 |
| `info` | 查询本机 Ascend NPU 设备信息 |
| `list` | 预览用例文件中的测试用例列表 |
| `help` | 命令帮助与参数深潜（`ttk help [COMMAND] [PARAM]`，亦可用 `ttk <mode> help [PARAM]`） |

查看版本：

```shell
python3 -m ttk -v
```

查看帮助：

```shell
python3 -m ttk kernel --help
python3 -m ttk aclnn --help
python3 -m ttk e2e --help
python3 -m ttk geir --help
```

# 设备信息与用例预览

> `ttk list` 仅列出用例名（含 `--op` 过滤、`--sheet` 选择、跳过 `is_enabled=false` 的行），除要求 `testcase_name` 列存在且非空（列缺失或该行值为空会直接报错）外不做其他字段校验；也不做表头体检与重名改写；完整字段校验请使用对应模式的 `--validate`。

```shell
# 查看设备信息
python3 -m ttk info

# 预览用例列表
python3 -m ttk list -i examples/case_store/kernel/cases.csv

# 按算子名过滤预览
python3 -m ttk list -i cases.csv --op add

# 预览 Excel 用例（指定工作表）
python3 -m ttk list -i cases.xlsx --sheet Sheet2
```

# 通用参数

| 参数 | 缩写 | 默认值 | 说明 | 示例 |
|------|------|--------|------|------|
| `--input` | `-i` | 无 | 用例文件路径，支持 csv/xlsx/xlsm/zip（必填）；zip 打包多个 csv 时一次运行展开为多张表 | `-i add.csv` 或 `-i cases.xlsx` |
| `--sheet` | | 首个工作表 | Excel 工作表名；仅 `.xlsx`/`.xlsm` 生效，csv 忽略。xlsx 默认输出名带实际 sheet 名（如 `cases_T2_result.csv`），多 sheet 互不覆盖 | `--sheet T2` |
| `--config` | | 无 | ttk 配置 YAML 路径（覆盖 `~/.config/ttk/` 和 `./ttk.conf.yaml`） | `--config ttk.conf.yaml` |

## 用例筛选

| 参数 | 缩写 | 默认值 | 说明 | 示例 |
|------|------|--------|------|------|
| `--testcase` | `-t` | 无 | 按用例名筛选（逗号分隔） | `-t add_01,add_02` |
| `--testcase-index` | `--ti` | 无 | 按用例索引筛选（从 0 开始，多表输入如 zip 打包多 csv 时索引按表独立编号） | `--ti=0,2,4` 或 `--ti=0-9` |
| `--testcase-count` | `--tc` | 无 | 随机选取N个用例 | `--tc=10` |
| `--operator` | `--op` | 无 | 按算子名筛选 | `--op add,mat_mul_v3` |
| `--exclude-operator` | `--no-op` | 无 | 排除算子名 | `--no-op concat_d` |
| `--priority` | | 无 | 按优先级筛选 | `--priority=1-3` |
| `--provider` | | 无 | 三方provider过滤（如torch/tf，别名 tensorflow→tf），缩小dispatch范围 | `--provider torch` |
| `--rerun` | | 无 | 重跑失败项（仅 Kernel/GEIR 通路支持；Kernel 可用列：`dyn_perf_us`/`cst_perf_us`/`bin_perf_us`/`perf_status`/`precision_status`，GEIR 可用列：`dyn_perf_us`/`cst_perf_us`/`precision_status`（`bin_perf_us`/`perf_status` 为 Kernel 专属结果列），ACLNN/E2E 传入会报 `Unsupported rerun targets`） | `--rerun=precision_status` |

## 设备与并行

| 参数 | 缩写 | 默认值 | 说明 | 示例 |
|------|------|--------|------|------|
| `--device` | `--dev` | 全部 | 使用设备数量 | `--dev 2` |
| `--devices` | | 无 | 设备 ID 表达式：`3` / `0,1` / `0-7` 正选 + `^` 剔除；可与 `--dev N` 组合（N 定池上限，正选池内取交集） | `--devices 0-7,^3` |
| `--process-count` | `--pc` | 自动（每卡 = CPU 核数 × 80% ÷ 可用卡数，限 1~4；`--backend npusim` 默认 1，显式 `--pc N` 可覆盖） | 每张卡进程数 | `--pc 4` |
| `--proc-no-reuse` | | 关闭 | 每个用例创建新进程 | `--proc-no-reuse` |
| `--no-memory-check` | | 关闭 | 跳过主机内存检查 | `--no-memory-check` |
| `--platform` | `--plat` | 自动检测 | SoC版本 | `--plat Ascend910B3` |
| `--proc-timeout` | | 3600 | 单用例超时（秒） | `--proc-timeout 120` |
| `--limit` | `-l` | 30 | 单用例HBM内存上限（GB），超过则判为无效用例（`STC_SHAPE_OUT_OF_BOUND`）：跳过编译/执行，但仍落一行失败结果（kernel/aclnn 通路消费；kernel 按输入+输出总量计，aclnn 仅按输入计） | `-l 16` |

## 执行控制

| 参数 | 缩写 | 默认值 | 说明 | 示例 |
|------|------|--------|------|------|
| `--run` | | 3（仿真模式 1） | 执行次数 | `--run=5` |
| `--task-prof` | | 开启 | 任务级 Profiling 开关 | `--task-prof=false` |
| `--warmup` | | 开启 | Profiling 前预热 | `--warmup=false` |
| `--npu-timeout` | | 无限制 | NPU 执行超时（ms），仅 kernel 通路（其余子命令无此参数） | `--npu-timeout 60000` |
| `--no-prof` | | 关闭 | 只生成输入和 Golden，不执行目标 API | `--no-prof` |
| `--validate` | | 关闭 | 仅校验 CSV 用例格式，不执行编译、输入生成、Golden 生成等流程 | `--validate` |

## 精度控制

| 参数 | 缩写 | 默认值 | 说明 | 示例 |
|------|------|--------|------|------|
| `--compare` | | 未指定 | 精度对比方法。可选：`close`/`stat_rel_err`/`mixed`/`cosine`/`binary`/`requant`/`cross_check`。未指定时按 Spec.tolerance 路由（需 `--plugin`）；无 Spec 声明时按输出 dtype 路由：浮点→`mixed`，整数/布尔→`binary`，complex→`close`，hifloat8→`requant`，float8_e8m0→`binary` | `--compare cosine` |
| `--input-dist` | | `uniform` | 输入数据分布，可选：`uniform`/`normal` | `--input-dist normal` |
| `--seed` | | 随机 | 随机种子：固定后同一份 CSV 连跑两次，每条用例的输入/golden 逐位一致（与进程数、执行顺序无关），可直接做改动前后的逐用例 A/B 对比；batch 一致性场景的同 shape 同值行为不受影响 | `--seed 42` |
| `--golden-mode` | | `Enable` | Golden生成模式，可选：`Enable`/`Disable`/`Promote`（CPU 高精度）/`XPU`（三方 XPU 输出作为 golden）。`Disable` 时不生成标杆、不做精度比对，用例精度结论为 `SKIP`（未验证，计入汇总的 SKIP/NA 桶，不计为通过） | `--golden-mode Disable` |
| `--deterministic-level` | `--dl` | `0` | 确定性等级：`0`=关闭，`1`=确定性计算，`2`=强一致，`3`=确定性执行；FA 完整 batch relation 的切片比对在第二阶段完成 | `--dl=1` |

> 精度比对方法的公式、容差及选择建议详见[精度比对方法](./Precision_Comparison.md)。确定性计算详见[确定性计算与批一致性](./Deterministic_Compute.md)。

## 调试与诊断

| 参数 | 缩写 | 默认值 | 说明 | 示例 |
|------|------|--------|------|------|
| `--dump` | | 关闭 | Dump数据：`full`/`in`/`out`/`golden`/`xpu`，可逗号组合；`full` 为全部四类（无远端 XPU 配置时自动跳过 XPU） | `--dump in,out,xpu` |
| `--dump-format` | | `bin` | Dump格式：`bin`/`npy`/`pt`/`print` | `--dump-format npy` |
| `--dump-on-fail` | | 关闭 | 精度失败时自动补 dump 输入/输出/Golden（不补 `xpu` 槽） | `--dump-on-fail` |
| `--manual-data-dirs` | | 无 | prepare输出目录或replay有序搜索目录 | `--manual-data-dirs /data/op` |
| `--plugin` | | 无 | 外部插件路径 | `--plugin /path/to/plugin.py` |
| `--xpu-perf` | | 关闭 | 采集三方XPU性能（需remote XPU配置） | `--xpu-perf` |

## 结果输出

| 参数 | 缩写 | 默认值 | 说明 | 示例 |
|------|------|--------|------|------|
| `--output` | `-o` | 无 | 输出结果CSV路径（覆盖已有文件），与 `-a` 互斥 | `-o results.csv` |
| `--append` | `-a` | 无 | 追加结果到已有CSV（表头不匹配则覆盖），与 `-o` 互斥 | `-a results.csv` |
| `--po` / `--progress-output` | | 无 | 进度输出路径（`Key:Value` 纯文本：StartAt/ElapsedTime/TotalCases/CompletedCases） | `--po /tmp/progress.txt` |

> 每个用例的独立日志目录是默认行为（`{root}/log/{mode}/{testcase_name}/ttk.log`），无需参数；摘要信息无条件打印，不可关闭。

# 通路专属参数

各通路专属参数详见对应指南：

- **Kernel编译控制**（`-d`/`-c`/`-b`/`--co`/`--compile-opts`等）：[Kernel测试指南](./Operator_Test_Guides/Kernel_Test_Guide.md)
- **E2E后端选择**（`--backend cpu`/Backend抽象层）：[E2E测试指南](./Operator_Test_Guides/E2E_Test_Guide.md)
- **GEIR编译模式**（`-c`/`-d`/`-b`）：[GEIR测试指南](./Operator_Test_Guides/GEIR_Test_Guide.md)
- ACLNN复用通用参数，无专属选项

# 高阶使用场景

| 文档 | 核心参数 |
|------|---------|
| [离线数据准备与导入](./Offline_Data_Prepare_and_Import.md) | `--no-prof --dump in,golden` + `--manual-data-dirs` |
| [XPU 三方交叉校验](./XPU_Cross_Check.md) | `--compare cross_check` / `--xpu-perf` + `--config` |
| [确定性计算与批一致性](./Deterministic_Compute.md) | `--deterministic-level` / `--dl` |
| [Dump 数据调试](./Dump_Debug.md) | `--dump` + `--dump-format` + `--dump-on-fail` |
| [性能测试](./Performance_Testing.md) | `--warmup` + `--run` |
| [NPUSim 仿真测试](./NPUSim/npusim_usage.md) | `--backend npusim` + `--sim-*` |
