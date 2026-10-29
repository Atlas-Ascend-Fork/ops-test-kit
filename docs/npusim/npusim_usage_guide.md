# TTK NPUSim 仿真测试使用指南

## 使用须知

通过 `--backend npusim`，让 TTK 的 **Kernel**、**ACLNN** 与 **E2E**（torch 域）测试模式在 NPUSim（昇腾 SoC 级仿真器）上执行，代替真实芯片，适用于**无卡 / 芯片资源紧缺**环境下的算子精度验证与性能分析。

- **仿真执行**：kernel 在周期精确芯片模型上运行，与真实芯片二进制兼容，同一 kernel 无需改写
- **精度比对**：golden 仍在 CPU 端生成，与仿真回读结果走 TTK 现有比对链路，结果照常写入结果 CSV
- **仿真流水图**：可选 `--sim-report` 生成指令流水图（Chrome tracing / Perfetto 可视化）

关联设计文档：`docs/npusim/npusim_design.md`

文档导览：

1. **[环境准备](#一环境准备)**
2. **[快速开始](#二快速开始)**
3. **[参数说明](#三参数说明)**
4. **[输出产物](#四输出产物)**
5. **[精度比对](#五精度比对)**
6. **[故障排查](#六故障排查)**
7. **[完整示例](#七完整示例)**

## 一、环境准备

### 1. 安装 CANN（需含 Ascend-950-ops 包）

```bash
./Ascend-cann-toolkit_<version>_linux-aarch64.run --install --install-path=<install_path>
./Ascend-cann-950-ops_<version>_linux-aarch64.run --install --install-path=<install_path>
```

> [!IMPORTANT]
> 必须安装 **Ascend-950-ops** 包：Ascend950 的 TBE 算子实现与 aclnn 算子库由它提供。缺包会导致 kernel 编译报 `OPTILING_FAILURE` / `*_OPERATOR_NOT_FOUND`，aclnn 执行报 `ACLNN_EXECUTE_FAILED`。

### 2. 使用 CANN 自带 cannsim

TTK 直接使用 CANN 自带的 `cannsim`（`$ASCEND_TOOLKIT_HOME/bin/cannsim`），无需另装 npu-simulator 仓库源码。

### 3. 加载环境

```bash
source <install_path>/cann/set_env.sh
```

## 二、快速开始

在 ops-test-kit 仓库根目录运行：

```bash
# Kernel / ACLNN
python3 -m ttk kernel --backend npusim -i examples/case_store/kernel/cases.csv -t add_fp16_broadcast --sim-cores 0
python3 -m ttk aclnn  --backend npusim -i examples/case_store/aclnn/cases.csv -t add_fp16_broadcast --sim-cores 0

# E2E（torch_npu，eager 执行）
python3 -m ttk e2e --backend npusim -i examples/case_store/e2e/torch_cases.csv -t add_f32_alpha

# 额外生成仿真流水图
python3 -m ttk kernel --backend npusim -i examples/case_store/kernel/cases.csv -t add_fp16_broadcast --sim-cores 0 --sim-report
```

> [!TIP]
> CSV 用例格式与真实芯片模式一致，仅在命令后追加 `--backend npusim`。

## 三、参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--backend {npu,npusim}` | `npu` | `npusim` 走仿真执行（e2e 另含 `cpu`） |
| `--sim-soc <SoC>` | `Ascend950` | 仿真 SoC，须为所装 cannsim `SOC_INFO` 中注册的版本 |
| `--sim-output <dir>` | `<root>/sim_output` | 仿真中间产物根目录 |
| `--sim-report` | 关闭 | 仿真结束后生成流水图（trace_core*.json + 报告） |
| `--sim-cores <范围>` | 全核 | 指定仿真核，如 `0-2,12-14` / `all` / `0`；建议设小范围以加快仿真 |
| `--sim-obj <kernel.o>` | 无 | 传给 report 的目标文件（含源码/汇编级信息） |

其他 TTK 常用参数（用例筛选、精度等）均可组合使用：`-t` / `--ti` / `--op` 等。

## 四、输出产物

```
<sim_output>/                          # 默认 sim_output/（--sim-output 可改）
├── wrappers/                          # 生成的 wrapper 脚本（.py）
└── <case_name>/                       # 每个用例一个目录
    ├── record_out/
    │   └── {cannsim,npusim}_<时间戳>_<用例>/
    │       ├── instr.bin              # 指令流（流水图输入）
    │       └── cannsim.log            # record 日志
    └── report/                        # 仅 --sim-report 时生成
        └── trace_core0.json           # 指令流水图（Chrome tracing）
```

**结果 CSV**：由 `-o` 指定（默认 `<输入名>_result.csv`），含 `precision_status`、`dyn_precision`、`memory_oob_status`、`soc` 等列。仿真模式不输出单次周期数：kernel 通路的 `dyn/cst/bin_perf_us` 为 `UNKNOWN`，aclnn 通路的 `api_perf_us`/`op_perf_us` 为空。

**查看流水图**：将 `trace_core0.json` 拖入 Chrome `chrome://tracing` 或 [Perfetto](https://ui.perfetto.dev)。

## 五、精度比对

- **golden**：CPU 端 numpy / torch 生成（`--golden-mode` 语义与真实模式一致）
- **容差**：沿用 TestSpec 的 `tolerance`（默认 `mixed`）
- **对比方法**：`--compare close|stat_rel_err|mixed|cosine|binary|requant|cross_check`
- **自定义 golden / 输入**：`--plugin <path>` 加载 TestSpec，与真实模式一致

## 六、故障排查

### 1. 运行约束

- 必须在仓库根目录执行 `python3 -m ttk ...`
- 必须 source 含 950-ops 的 CANN 环境，否则算子编译/执行失败
- 仿真耗时较长：初始化约 20–30 秒，大 shape 用例可达几十分钟；建议 `--sim-cores 0` + 小 shape 先行验证
- 并发进程默认 1（避免多仿真实例同时运行耗尽 CPU）
- E2E 仿真仅覆盖 torch 域（onnx 不支持 `--backend npusim`，启动期报 `ONNX_NPUSIM_UNSUPPORTED`）；建议仅用 eager，部分算子在仿真下可能挂起，先小 shape 验证算子受支持

### 2. 故障排查

| 现象 | 可能原因 | 排查 |
|---|---|---|
| `OPTILING_FAILURE` / `*_OPERATOR_NOT_FOUND` | 缺 950-ops 包或算子 TBE 实现 | 确认已装 `Ascend-cann-950-ops`；看各用例日志 `ttk.log` |
| `ACLNN_EXECUTE_FAILED` | aclnn 算子库不支持该 SoC | 同左，确认 950-ops 已装 |
| `SIM_RESULT_MISSING` / `wrapper_error.json` 有内容 | wrapper 执行异常 | 看 `<sim_output>/<case>/wrapper_error.json` 与 `cannsim.log` |
| `RuntimeError: Current device only support aclnn operator` | 用例调用了非 aclnn 的旧版自定义算子 | 换用有 aclnn 实现的算子，或改真机 `--backend npu` 验证 |
| `precision_status: FAIL` | golden 与仿真输出偏差 | 检查容差、golden 实现、shape/format 定义 |
| report 报 `No executed instructions` | 用错 cannsim 版本（格式不匹配） | 确认 `--sim-report` 走环境自带 `$ASCEND_TOOLKIT_HOME/bin/cannsim` |
| `CANN cannsim not found` | 未安装含 cannsim 的 CANN | 确认 `$ASCEND_TOOLKIT_HOME/bin/cannsim` 存在 |
| 用例卡死在 `OnDynProfiling` | 个别用例组合的仿真收尾忙自旋 | 已内置规避，无需干预 |

### 3. 已知限制

- 仅支持所装 cannsim `SOC_INFO` 中注册的 SoC（如 `Ascend950` / `Ascend950DT`）
- 仿真性能数据为 `UNKNOWN`（无单值周期），性能分析依赖 `--sim-report` 流水图
- 需 numpy / plotly（流水图用）等 Python 依赖

## 七、完整示例

```bash
source <install_path>/cann/set_env.sh
cd <ops-test-kit 仓库路径>

# 单用例 + 核0 + 流水图 + 指定结果文件
python3 -m ttk kernel --backend npusim \
    -i examples/case_store/kernel/cases.csv -t add_fp16_broadcast \
    --sim-cores 0 --sim-report \
    -o add_npusim_result.csv
```

预期输出（节选，收尾汇总）：

```
Total  PASS  FAIL  SKIP/NA  PassRate
1      1     0     0        100.00% (1/1)
```

产物：

- 结果：`add_npusim_result.csv`
- 流水图：`sim_output/add_fp16_broadcast/report/trace_core0.json`
