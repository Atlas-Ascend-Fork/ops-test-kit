# TTK NPUSim 仿真测试使用指南

> 通过 `--backend npusim`，让 TTK 的 **Kernel**、**ACLNN** 与 **E2E** 测试模式在 NPUSim（昇腾 SoC 级仿真器）上执行，代替真实芯片。适用于**无卡 / 芯片资源紧缺**环境下的算子精度验证与性能分析。
> 关联设计文档：`docs/NPUSim/npusim_design.md`

---

## 1. 原理简介

- **仿真执行**：TTK 生成一个 Python wrapper 程序，由 NPUSim `record` 在 camodel（周期精确芯片模型）中拉起执行——与真实芯片**二进制兼容**，同一 kernel 无需改写。
- **精度比对**：golden 仍在 CPU 端（numpy/torch）生成，与仿真回读的设备输出走 TTK 现有比对链路，结果照常写入结果 CSV 的 `precision_status` 等列。
- **仿真流水图**：可选 `--sim-report`，调用环境自带 `cannsim report` 从仿真的 `instr.bin` 生成指令流水图（Chrome tracing / Perfetto 可视化）。

```
CSV 用例 ──► TTK 编译 + CPU golden ──► NPUSim record（wrapper 在 camodel 中执行）
                                                        │
                                   仿真输出 ──► 精度比对（precision_status）
                                   instr.bin ──► report（trace_core*.json 流水图）
```

---

## 2. 环境准备（一次性）

### 2.1 安装 CANN（需含 Ascend-950-ops 包）

```bash
./Ascend-cann-toolkit_<version>_linux-aarch64.run --install --install-path=<install_path>
./Ascend-cann-950-ops_<version>_linux-aarch64.run --install --install-path=<install_path>
```

> 必须安装 **Ascend-950-ops** 包：Ascend950 的 TBE 算子实现（如 `add_apt.py`）与 aclnn 算子库（`libopapi*.so`）由它提供。缺包会导致 kernel 编译报 `OPTILING_FAILURE` / `*_OPERATOR_NOT_FOUND`，aclnn 执行报 `ACLNN_EXECUTE_FAILED`。

### 2.2 使用 CANN 自带 cannsim

TTK 通过命令行调用 NPUSim，直接使用 **CANN 自带的 `cannsim`**（`$ASCEND_TOOLKIT_HOME/bin/cannsim`，与已安装的 camodel 同版本配套），**无需另装 npu-simulator 仓库源码**。安装含 cannsim 的 CANN 后即可使用。

### 2.3 加载环境

建议每次运行前先 source 新环境的 set_env（TTK 会继承 `ASCEND_TOOLKIT_HOME` 等变量；未 source 时 TTK 启动会尝试自动 source 并打 warning）：

```bash
source <install_path>/cann/set_env.sh
```

---

## 3. 快速开始

在 **ops-test-kit 仓库根目录**运行（TTK 依赖仓库内相对路径）：

```bash
# Kernel 模式：编译 + 仿真执行 + 精度比对
python3 -m ttk kernel --backend npusim -i examples/case_store/kernel/cases.csv -t add_fp16_broadcast --sim-cores 0

# ACLNN 模式
python3 -m ttk aclnn --backend npusim -i examples/case_store/aclnn/cases.csv -t add_fp16_broadcast --sim-cores 0

# E2E 模式（torch_npu，eager 执行）
python3 -m ttk e2e --backend npusim -i examples/case_store/e2e/torch_cases.csv -t add_f32_alpha

# E2E 模式 + 额外生成仿真流水图
python3 -m ttk e2e --backend npusim -i examples/case_store/e2e/torch_cases.csv -t add_f32_alpha --sim-report

# 额外生成仿真流水图（Kernel / ACLNN）
python3 -m ttk kernel --backend npusim -i examples/case_store/kernel/cases.csv -t add_fp16_broadcast --sim-cores 0 --sim-report
```

CSV 用例格式与真实芯片模式**完全一致**，无需增加任何字段，仅在命令后追加 `--backend npusim`。

---

## 4. 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--backend {npu,npusim}` | `npu` | `npusim` 走仿真执行（e2e 另含 `cpu`） |
| `--sim-soc <SoC>` | `Ascend950` | 仿真 SoC，须为所装 cannsim `SOC_INFO` 中注册的版本（如 `Ascend950` / `Ascend950DT` / `Ascend960`） |
| `--sim-output <dir>` | `<root>/sim_output` | 仿真中间产物根目录 |
| `--sim-report` | 关闭 | 仿真结束后生成流水图（trace_core*.json + 报告） |
| `--sim-cores <范围>` | 全核 | 指定仿真核，如 `0-2,12-14` / `all` / `0`。**建议设小范围以加快仿真** |
| `--sim-obj <kernel.o>` | 无 | 传给 record `-f` 的目标文件（报告含源码/汇编级信息） |

其他 TTK 常用参数（用例筛选、精度等）均可组合使用：

```bash
# 筛选单个用例、指定核、输出流水图
python3 -m ttk kernel --backend npusim -i examples/case_store/kernel/cases.csv -t add_fp16_broadcast --sim-cores 0 --sim-report -o result.csv

# 按索引 / 算子筛选
python3 -m ttk kernel --backend npusim -i cases.csv --ti=0-2 --sim-cores 0
python3 -m ttk aclnn  --backend npusim -i cases.csv --op add   --sim-cores 0
```

---

## 5. 输出产物

运行后终端输出每个用例的精度结果与整体通过率；同时产出：

```
<sim_output>/                          # 默认 sim_output/（--sim-output 可改）
├── wrappers/                          # 生成的 wrapper 脚本（.py）
└── <case_name>/                       # 每个用例一个目录
    ├── dyn/cst/bin 或 aclnn 的执行参数与输出  # kernel 按编译模式分目录、aclnn 序列化 context；含回读输出
    ├── record_out/
    │   └── {cannsim,npusim}_<时间戳>_<用例>/   # 前缀随 CANN 版本（旧版 cannsim_，9.2.0+ npusim_）
    │       ├── instr.bin              # 指令流（流水图输入，位于归档根目录）
    │       └── cannsim.log            # record 日志
    └── report/                        # 仅 --sim-report 时生成
        └── trace_core0.json           # ★ 指令流水图（Chrome tracing）
```

**结果 CSV**：由 `-o` 指定（默认 `<输入名>_result.csv`），含 `precision_status`、`dyn_precision`、`memory_oob_status`、`soc` 等列。仿真模式下性能无单值周期（性能请查看流水图）：kernel 通路 `dyn/cst/bin_perf_us` 列为 `UNKNOWN`；aclnn 通路 `api_perf_us`/`op_perf_us` 为空、`perf_status` 为 `UNKNOWN`。

**查看流水图**：将 `trace_core0.json` 拖入 Chrome `chrome://tracing` 或 [Perfetto](https://ui.perfetto.dev)。

---

## 6. 精度比对

- **golden**：CPU 端 numpy / torch 生成（`--golden-mode` 语义与真实模式一致）。
- **容差**：沿用 TestSpec 的 `tolerance`（默认 `mixed`）；CSV 的 `precision_tolerances` / `absolute_precision` 为 legacy 字段，`precision_tolerances` 由 `close`/`cosine`/`requant` 读取、`absolute_precision` 仅 `close` 读取。
- **对比方法**：`--compare close|stat_rel_err|mixed|cosine|binary|requant|cross_check`（默认按 Spec.tolerance 路由）。
- **自定义 golden / 输入**：`--plugin <path>` 加载 TestSpec，与真实模式一致。

---

## 7. 注意事项与故障排查

### 7.1 运行约束

- **必须在仓库根目录执行** `python3 -m ttk ...`。
- **必须 source 新 CANN 环境**（含 950-ops），否则算子编译/执行失败。
- **仿真耗时较长**：camodel 初始化约 20–30 秒；大 shape 用例（如 `(2576, 8104)`）可达几十分钟。建议 `--sim-cores 0` + 小 shape 先行验证。
- 并发进程默认自动设为 1（避免多 camodel 打爆 CPU）。

### 7.2 故障排查

| 现象 | 可能原因 | 排查 |
|---|---|---|
| `OPTILING_FAILURE` / `*_OPERATOR_NOT_FOUND` | 缺 950-ops 包或算子 TBE 实现 | 确认已装 `Ascend-cann-950-ops`；看 per-case 日志目录下的 `ttk.log` |
| `ACLNN_EXECUTE_FAILED` | aclnn 算子库不支持该 SoC | 同左，确认 950-ops 已装 |
| `SIM_RESULT_MISSING` / `wrapper_error.json` 有内容 | wrapper 执行异常 | 看 `<sim_output>/<case>/wrapper_error.json` 与 `record_out/cannsim_*/cannsim.log` |
| `RuntimeError: Current device only support aclnn operator` / `ERR00007 PTA feature not supported` | 用例调用了非 aclnn 的 legacy 自定义算子（如 `torch_npu.npu_conv2d`），camodel 仅支持 aclnn 算子 | 换用有 aclnn 实现的算子，或该算子改用真机 `--backend npu` 验证 |
| `precision_status: FAIL` | golden 与仿真输出偏差 | 检查容差、golden 实现、shape/format 定义 |
| report 报 `No executed instructions` | 用错 cannsim 版本（格式不匹配） | 确认 `--sim-report` 走环境自带 `$ASCEND_TOOLKIT_HOME/bin/cannsim` |
| `CANN cannsim not found` | 未安装含 cannsim 的 CANN，或 `cannsim` 不在 PATH | 确认已装 CANN 且 `$ASCEND_TOOLKIT_HOME/bin/cannsim` 存在 |
| 用例卡死：`OnDynProfiling` 长时间 `RUNNING`，`cannsim.log` 停在 `TASK_BEGIN RTSQ_1` 无 `TASK_DONE` | Ascend950 camodel teardown（`rtCtxDestroy`）对个别用例组合忙自旋 | 已内置规避：wrapper 通过 `skip_teardown=True` 跳过 camodel teardown，无需干预 |

### 7.3 已知限制

- 仅支持所装 cannsim `SOC_INFO` 中注册的 SoC（如 `Ascend950` / `Ascend950DT` / `Ascend960`）。
- 仿真性能数据为 `UNKNOWN`（无单值周期），性能分析依赖 `--sim-report` 的流水图。
- 需 numpy / plotly（流水图用）等 Python 依赖。
- **wrapper 跳过 camodel teardown**：Ascend950PR_9589 camodel 的 `rtCtxDestroy` / `rtDeviceReset` 对个别用例组合存在忙自旋缺陷（执行成功后 teardown 卡死）。TTK 的仿真 wrapper 通过 `skip_teardown=True` 跳过 teardown 调用——wrapper 以 `os._exit` 结束进程，进程退出即回收 camodel 资源，故该跳过对精度结果无影响（已实证输出 bit 级正确）。
- **E2E 模式（`--backend npusim`）**：
  - 聚焦 **eager** 执行；graph 模式（cst/dyn/aclgraph）依赖 `torchair` 且可能走 GE 图编译（camodel 缺 pcie bar 能力），仿真下未验证、可能失败，建议仅用 eager。
  - **算子支持受 camodel / Ascend950 ops 集限制**：已实证 `torch.add`（含 f16/f32/广播）可跑通；部分算子（如 `torch.abs`、`torch.Tensor.relu_` inplace）在 camodel 下会**挂起**（`OnEagerProfiling` 长时间 RUNNING）。**camodel 仅支持 aclnn 算子**：非 aclnn 的 legacy 自定义算子（如 `torch_npu.npu_conv2d`）执行时被 torch_npu 直接拒绝，报 `RuntimeError: Current device only support aclnn operator, but current operator xxx do not have aclnn implementation`（`ERR00007 PTA feature not supported`）。跑仿真前请先用小 shape 单用例确认算子受支持。
  - eager `*_device_perf_us` 为 `----`（camodel 无单值周期），精度比对不受影响。
  - **支持 `--sim-report`**：camodel 把指令轨迹写为 `instr.bin`，TTK 每用例执行后归档到 `<sim_output>/<case>/instr.bin`（E2E 仿真直跑 torch_npu，不经 cannsim record，无 record_out 层级），并生成 `<sim_output>/<case>/report/trace_core*.json`（Chrome Tracing 流水图，可加载到 `chrome://tracing` / Perfetto 查看）。
  - 环境注入：TTK 在 FrameworkApiInstance 启动时自动把 camodel 目录前置到 `LD_LIBRARY_PATH`（profiling worker 经 forkserver 继承），无需手工设置。

---

## 8. 示例：完整跑通 add

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
