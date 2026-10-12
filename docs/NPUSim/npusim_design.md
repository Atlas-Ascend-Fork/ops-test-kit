# TTK × NPUSim 仿真测试集成设计文档

> record/report 使用 CANN 自带 cannsim，不依赖 npu-simulator 仓库源码。
> 关联仓库：`ops-test-kit`（TTK）
>
> **使用者请参阅**：[TTK NPUSim 使用指南](./npusim_usage.md)（命令、参数、排障）

## 1. 背景与目标

TTK（`ops-test-kit`）目前只能在真实 Ascend 芯片上执行算子。NPUSim（`npu-simulator`，SoC 级仿真工具）能在无卡环境下以二进制兼容方式运行同一 kernel，产出 bit 级精度结果与指令流水图。

**目标**：让 TTK 的 **kernel**、**aclnn** 与 **e2e** 测试模式可选地通过 NPUSim 仿真执行代替真实芯片（geir 仅有 NPU 图形态，不接仿真后端），输出：

1. TTK 现有精度比对结果（复用 golden + comparison 体系 → 结果 CSV 的 `precision_status` 等列）
2. 仿真流水图（复用 NPUSim `report` 的 `trace_core*.json` Chrome tracing + HTML 报告，显式开关触发）

**已确认决策**：

1. 集成形态：`ttk kernel` / `ttk aclnn` / `ttk e2e` 命令内加 `--backend npusim` 开关切换执行后端。
2. wrapper（user_app）形式：**Python wrapper**（NPUSim 原生支持 `.py`，自动注入 `TASK_QUEUE_ENABLE=0`）。
3. 流水图：默认不生成，`--sim-report` 显式触发。

## 2. 关键事实（已实证）

### NPUSim 侧

| 事实 | 说明 |
|---|---|
| user_app 直接给 `.py` 路径即可 | `launch_spec.py` 判非 ELF → `bash -c` 执行，`is_python=True` 注入 `TASK_QUEUE_ENABLE=0`；退出码 0 = PASS（`runner.py:179`） |
| record 持久产物仅 `instr.bin` + `.soc-version` | `log/`、`log_ca/`、`run_log/` 跑完被 `artifacts.py::cleanup_internal_dirs` 清理 |
| **NPUSim 不产出设备端 tensor 数值** | 必须由 user_app（TTK 生成的 wrapper）自行回读算子输出并写盘，TTK 再读取比对 |
| report 输入优先级 | helper DB → `instr.bin` → `log_ca/`；引擎 BiProfRunner（需 plotly≥5） |
| report 产物 | `trace_core{N}.json`（Chrome tracing）、`kernel_<N>_reports/core_<M>/` 下各 `*_report.html`、聚合 `index.html`/`SUMMARY.md`、`*_vf_report.json` |
| 环境依赖 | `ASCEND_TOOLKIT_HOME`（需 source setenv）、camodel 库（toolkit 自带，record 注入 `LD_LIBRARY_PATH`） |
| SoC 映射 | `Ascend950 → Ascend950PR_9589`；`SOC_INFO` max_cores=32 |
| **版本兼容** | CANN 9.1.0-beta.3 之前自带的 `cannsim` instr.bin 记录格式不兼容（424→432 字节）；9.1.0-beta.3 / 9.2.0 自带 cannsim（v0.1.0）与 camodel 配套、行为一致（record/report 全参数、`cannsim_` 产物前缀），record/report 均直接走 `$ASCEND_TOOLKIT_HOME/bin/cannsim`，无需 npu-simulator 仓库源码 |

### TTK 侧

- CLI 分发：`ttk/cli/kernel.py|aclnn.py` → `bridge.py::args_to_switches()` 灌入 `SWITCHES` → `run_with_switches()` → 通路 Instance（`KernelInstance` / `AclnnInstance`）。
- **执行接缝点**：
  - kernel：`kernel/execute.py` `execute_on_device()` → `KernelExecutor.execute()`（`kernel/execute.py`）→ `RTSInterface`（`runtime/rts_interface.py`，ctypes 加载 libruntime.so）→ `rtKernelLaunchWithFlagV2`。返回 `KernelExecuteResult(cycle, output_bytes, oob)`。
  - aclnn：`aclnn/execute.py` `execute_on_device()` → `AclOpExecutor._acl_sequence()` → `AclInterface`（`aclnn/runtime.py`，libascendcl.so + libnnopbase.so）→ `aclnnXxxGetWorkspaceSize` + `aclnnXxx`。返回 `ApiProfilingResult(success, api_prof, op_prof, output_bytes, output_view_shapes)`。
- 比对：`common/comparison/comparison.py` `compare(outputs, goldens, dtypes, standards)`；选择逻辑 `common/comparison/resolve.py`；结果 `EachCompareResult(precision, is_pass, log, standard, metrics)`。
- **已有 model 语义可复用**：`MODE.ASCEND_CAMODEL`（`ttk/switches.py`）时 `use_device()=False`、`run_time=1`、`RTSInterface(camodel=True)` 加载 `libruntime_camodel.so`、跳过 clear_l1/ub、`rts_stream=None`。
- Manual Data 两阶段：`--no-prof --dump in,golden` prepare / `--manual-data-dirs` replay（`bridge.py::configure_manual_data`、`ttk/core_modules/common/dump/manual_data.py`），现行文件命名 `input_{idx}_{dtype}.{ext}`、`golden_{idx}_{dtype}.{ext}`（`__shape_{dims}` 段仅为存量 bin Golden 兼容），支持 bin/npy/pt。

## 3. 总体架构

新增 `ttk/core_modules/common/simulator/`。**复用优先，不另起炉灶**：

- wrapper 直接 import TTK 现有 `RTSInterface` / `KernelExecutor` / `AclInterface`（纯 Python+ctypes，可放进子进程脚本）。
- 结果回填复用现有比对与结果 CSV 链路。
- 目录命名贴近 `manual_data.py` 规范。

```
ttk/core_modules/common/simulator/
├── __init__.py          # 导出 run_kernel_sim / run_aclnn_sim / case_dir / maybe_generate_sim_report / resolve_platform_soc / resolve_camodel_lib_dir
├── config.py            # SIM_PLATFORM_BY_SOC SoC 注册表 + 平台/camodel 路径解析（仿真参数默认值存于 SWITCHES sim_* 字段，经 sim_args.apply_sim_args 灌入）
├── npusim_runner.py     # 定位并调用 CANN 自带 cannsim：record/report 子进程封装、env 组装
├── case_writer.py       # 把 worker 内存中的输入/golden/spec 落盘为 <sim_output>/<case>/
├── wrapper.py           # 生成 kernel/aclnn 两种 wrapper .py（字符串模板）
├── sim_profiling.py     # run_kernel_sim(context) / run_aclnn_sim(context, dev_id) 主逻辑
└── report.py            # --sim-report 时调用 npusim report
```

## 4. 关键改动

### 4.1 SWITCHES 新字段（`ttk/switches.py`）

- `__slots__` 新增：`backend`、`sim_soc_version`、`sim_output_dir`、`sim_report`、`sim_cores`、`sim_object_file`。
- 默认值：`backend="npu"`、`sim_soc_version="Ascend950"`、`sim_output_dir=""`（空 → `root_path/sim_output`）、`sim_report=False`。
- 补 `__getstate__/__setstate__`（遍历 slots），保证 SWITCHES 可 pickle 进 wrapper。

### 4.2 CLI 参数（新增 `ttk/cli/sim_args.py`，kernel.py / aclnn.py / e2e.py 调 `add_sim_args(parser)`；geir.py 仅注册 `--backend`（npu），无仿真参数）

```
--backend {npu,npusim}       # 默认 npu（e2e 另有 cpu）
--sim-soc <Ascend950>     # 默认 Ascend950
--sim-output <dir>        # 仿真中间产物根目录（默认 root_path/sim_output）
--sim-report              # 仿真结束后生成流水图报告
--sim-cores <0-2,5>       # 传给 record/report 的 -n（默认空=全核）
--sim-obj <kernel.o>      # 可选，传给 record -f，报告含源码级信息
```

### 4.3 sim_args 接线（`ttk/cli/sim_args.py::apply_sim_args`）

- 各 mode handler 在 `args_to_switches` **之后**调用 `apply_sim_args(sw, args)`（不在 `args_to_switches` 内）：把 `--backend/--sim-*` 灌入 SWITCHES，`backend=="npusim"` 时做"仿真语义归一"：
  - `sw.mode = MODE.ASCEND_CAMODEL`（复用 model 语义：`use_device=False`、`run_time=1`、`rts_stream=None`、跳过 clear_l1/ub）
  - `sw.dev_plat = resolve_platform_soc(sw.sim_soc_version)`（把 SoC 名映射为平台 ini 名，如 `Ascend950 → Ascend950PR_9589`）
  - `sw.warmup = False`、`sw.task_profiling = False`、`sw.deterministic_level = 0`、默认 `sw.process_per_device = 1`（防多 camodel 打爆 CPU；显式 `--pc N` 可覆盖）
  - 与 `--no-prof` 互斥校验（仿真后端自行准备输入/golden）
- `configure_manual_data` **不改**：`--backend npusim` 不带 `--no-prof/--manual-data-dirs` 时它早返回，与透明仿真路径不冲突。

### 4.4 KernelInstance 平台解析（`ttk/core_modules/kernel/instance.py::detect_platform`）

- 无仿真短路分支：CLI 层已把 `dev_plat` 预映射为平台 ini 名（如 `Ascend950PR_9589`，`SIM_PLATFORM_BY_SOC`），`detect_platform` 走正常 `get_npu_hw_info(dev_plat)` 路径解析 `short_soc_version`/ccec 版本（CANN 平台表含 `Ascend950PR_9589.ini`）。
- `setup()`：`_compile_help_kernels()` 增加 `and sw.backend != "npusim"`（仿真不编译 warmup/clear 辅助核）。

### 4.5 执行接缝分支（核心）

**kernel** — `_device_profiling()`（`kernel/worker.py`）三连调处改为：

```python
if switches.backend == "npusim":
    (context.dyn_prof_result, context.cst_prof_result,
     context.bin_prof_result) = run_kernel_sim(context)
else:
    # 真机执行三连调（不动）
```

**aclnn** — `_device_profiling()`（`aclnn/worker.py`）改为：

```python
if switches.backend == "npusim":
    context.prof_result = run_aclnn_sim(context, dev_id)
else:
    context.prof_result = execute_on_device(context, dev_id)
```

- `run_kernel_sim` 返回三个 `KernelExecuteResult(cycle=<wrapper result.json 的 cycle 值>, output_bytes=[bytes...], oob="OK")`（camodel 模式 cycle 为 `"UNKNOWN"`，`handle_profiling_result` 对 model 模式未测量值按 PASS 放行；真实性能在 report，后续可解析回填）。
- `run_aclnn_sim` 返回 `ApiProfilingResult(success=True, api_prof="UNKNOWN", op_prof="TOTAL_CYCLE_TODO", output_bytes=[...], output_view_shapes=[...], oob="OK")`。

### 4.6 Python wrapper

- 生成时机：`run_kernel_sim`/`run_aclnn_sim` 每次执行时写入 `<sim_out>/wrappers/<case>_<mode>.py`。
- 拉起：`npusim record` 的 user_app 直接给 wrapper 的**绝对路径**。
- wrapper 进程内 `PYTHONPATH` 前置 `sw.root_path`（TTK 源码，继承 CANN env；不含 npu-simulator 仓库），直接 `from ttk.core_modules.runtime import RTSInterface` 等，**不复制 ctypes 代码**。
- wrapper 收参：`sys.argv[1]` = 用例目录绝对路径。

**kernel wrapper**：反序列化 `switches.pkl` + 每模式 `param.pkl`（`RTSProfilingParam`，父进程 `construct_profiling_param` 构造后 pickle；`param.kernel_dir` 绝对路径指向 `kernel_meta`）→ `device = RTSInterface(camodel=True)` → 依序执行启用的 dyn/cst/bin（`KernelExecutor(device, param).execute()`）→ 每个输出写 `output_{i}.bin` + `result.json`（cycle/oob/ok）。

**aclnn wrapper**：`AclInterface` 可直接 import 复用；case 目录落盘 `switches.pkl` + `context.pkl`（整个 `TestcaseAclnn`），wrapper 直接复用 `Phase1ParamBuilder`/`AclOpExecutor` 执行：`AclOpExecutor(context, dev).do()` → 对 `output_tensor_indexes` 回读写 `output_{i}.bin` + `output_view_shapes.pkl` + `final_result.json`（ok/api_prof/op_prof/oob）。**补单测**：对样例 op 断言 wrapper 参数构建与 `Phase1ParamBuilder` 一致，防逻辑漂移。

## 5. 数据流（`ttk kernel --backend npusim -i cases.csv`）

1. CLI → SWITCHES（backend=npusim, mode=CAMODEL）→ `KernelInstance`（平台短路、device_count=1）→ worker 池。
2. worker `run()`：正常生成输入、golden（CPU）、解析 tolerance——全部复用现有逻辑。
3. `run_kernel_sim(context)`：
   - 启用的 dyn/cst/bin 各构造 `RTSProfilingParam` → pickle 落盘 `<sim_out>/<case>/{switches.pkl,dyn/,cst/,bin/}`（未启用模式父进程置 `KernelExecuteResult.fail("SUPPRESSED")`）。
   - 生成 wrapper.py → `npusim_runner.run_record`（一次 record 跑全部启用模式，wrapper 依序执行并写 `output_*.bin`/`result.json`）。
   - 读回 → `KernelExecuteResult`。
4. `handle_profiling_result` + `comparing()` → `KernelReturnStructure` → 结果 CSV（`precision_status` 等列照常产出）。
5. `--sim-report` 时 `npusim report` 生成流水图（独立产物，失败仅 WARN，不阻断精度）。

## 6. 目录规范（一个用例）

```
<sim_output>/                      # sw.sim_output_dir
├── wrappers/
│   └── caseA_kernel.py / caseA_aclnn.py
└── caseA/
    ├── switches.pkl               # pickle SWITCHES（kernel/aclnn 共用）
    ├── dyn/ cst/ bin/             # kernel：param.pkl + output_*.bin + result.json
    ├── context.pkl                # aclnn：整个 TestcaseAclnn pickle
    ├── output_*.bin + output_view_shapes.pkl + final_result.json   # aclnn 回读产物
    └── record_out/
        └── {cannsim,npusim}_<ts>_caseA/    # 前缀随 CANN 版本（旧版 cannsim_，9.2.0+ npusim_）
            ├── instr.bin, .soc-version, cannsim.log   # instr.bin 在归档根目录（CANN cannsim 布局）
            └── (report/ 仅 --sim-report)
```

## 7. npusim 调用封装（`simulator/npusim_runner.py`）

- 调用形态：`python3 $ASCEND_TOOLKIT_HOME/bin/cannsim record <wrapper绝对路径> -s <soc> -o <case_dir>/record_out [-u <case_dir>] [-n <cores>] [-f <obj>]`——用 `sys.executable` 执行 CANN 自带 cannsim 入口脚本（与 camodel 同版本配套），不再依赖 npu-simulator 仓库源码。
- `locate_cannsim_executable`：优先 `$ASCEND_TOOLKIT_HOME/bin/cannsim`，其次 PATH 上的 `cannsim`；找不到抛 `RuntimeError`（提示安装含 cannsim 的 CANN）。
- env：继承 TTK 已 source 的 CANN 环境（`PYTHONPATH` 仅前置 `sw.root_path`，不含仓库）；设 `NPUSIM_NO_DELAY=1`/`CANNSIM_NO_DELAY=1` 跳过 record 阶段间 sleep。
- PASS 判定：**以 wrapper 写的 `result.json` 为准**（record 退出码因 camodel teardown 不可靠）；超时用 `sw.proc_timeout` 兜底。
- `run_report`：`-e <record_out/cannsim_*/> -o <case_dir>/report -n <cores>`；`--sim-report` 时调用。

## 8. 与 Manual Data 的整合

**透明模式**：用户只传 `--backend npusim` 即可跑（内部自动完成输入/golden 生成 → 序列化 → 仿真 → 回填），无需先跑 prepare。

兼容：若用户额外传 `--manual-data-dirs`，则走"外部 prepare + 仿真 replay"。`--backend npusim --no-prof` 校验互斥并报错提示。

## 9. 多用例/批量

- 每用例独立一次 record（kernel 不同无法合并，合并会使 instr.bin 混叠）。
- 沿用 TTK worker 池，但 `apply_sim_args` 默认 `process_per_device=1`（显式 `--pc N` 可覆盖）；`--proc-timeout` 透传防卡死。
- 仿真慢（单用例秒级到分钟级）→ 三 kernel 模式合并一次 record 缓解。

## 10. 风险与验证

**最高优先级风险**：TBE 编译产物能否在 Ascend950 仿真直跑（TTK 编译按 `dev_plat` 选 arch，Ascend950 平台表无 ini）。**第一步原型验证**：用手写最小 add 用例跑 `--backend npusim`，确认 `rtKernelLaunchWithFlagV2` 在 camodel 上可执行；若 arch 不匹配需 TTK 编译侧接受 Ascend950（唯一可能触及编译链路的点）。

| # | 风险 | 缓解 |
|---|---|---|
| 1 | TBE 编译产物能否在 Ascend950 仿真直跑 | 最高优先原型验证；必要时编译侧透传 Ascend950 |
| 2 | aclnn 算子 so 依赖 ops 包 | 缺失时返回 `ApiProfilingResult.fail("OP_API_NOT_FOUND")` 并提示装 ops 包 |
| 3 | report 需 plotly + CANN 自带 cannsim | `--sim-report` 走 `$ASCEND_TOOLKIT_HOME/bin/cannsim`（与 camodel 配套）；失败仅告警 |
| 4 | `RTSInterface(camodel=True)` 若 camodel 缺符号 | 回退 `camodel=False`（经 LD_LIBRARY_PATH 加载 camodel 的 libruntime.so shim） |
| 5 | 仿真无单值 cycle | 当前置 1.0（PASS）；性能分析用 report 流水图 |
| 6 | aclnn wrapper 与 `Phase1ParamBuilder` 逻辑漂移 | 补参数构建对齐单测 |

## 11. 验证方式

1. **单元测试**：SWITCHES pickle 往返、sim 参数灌入、`apply_sim_args` 归一语义、wrapper 模板生成、aclnn wrapper 参数构建对齐 `Phase1ParamBuilder`。
2. **最小原型（最高优先）**：`python3 -m ttk kernel --backend npusim -i examples/case_store/kernel/cases.csv -t add_fp16_broadcast` 跑通 add，确认 camodel 可执行、输出回读、precision_status 产出。
3. **aclnn 原型**：`python3 -m ttk aclnn --backend npusim -i examples/case_store/aclnn/cases.csv -t add_fp16_broadcast`（需 ops 包）。
4. **流水图**：加 `--sim-report`，验证 `<case_dir>/report/trace_core*.json` 与 `index.html` 生成。
5. **批量/回归**：多用例 CSV + `--pc 1` 跑通，结果 CSV 各列正常。

## 12. 涉及文件

**simulator 集成**：`ttk/core_modules/common/simulator/{__init__,config,npusim_runner,case_writer,wrapper,sim_profiling,report}.py`、`ttk/cli/sim_args.py`、`tests/core_modules/common/simulator/test_simulator_backend.py`
**相关模块**：`ttk/switches.py`（SWITCHES 字段 + pickle）、`ttk/cli/kernel.py`、`ttk/cli/aclnn.py`、`ttk/cli/e2e.py`、`ttk/core_modules/kernel/instance.py`、`ttk/core_modules/kernel/worker.py`、`ttk/core_modules/aclnn/worker.py`、`ttk/core_modules/aclnn/input_generate.py`

## 13. 实证结论与已知问题

### 关键实证结论

1. **camodel teardown SIGSEGV（进程级）**：用户程序跑完后 `atexit`/reset 触发 camodel 清理崩溃（exit=-11），`try/except` 捕获不了。**wrapper 一律以 `os._exit(0)` 收尾**（绕过 atexit），且 **PASS/FAIL 判定以 wrapper 写的 `result.json` 为准，不依赖 record 退出码**。
2. **`record` 以 user_app 脚本目录为 cwd**（`launch_spec._build_file_launch_spec`），wrapper 与 case 路径必须绝对路径；`-u <case_dir>` 传给 wrapper 的 `sys.argv[1]`。
3. **`record` 持久产物仅 `instr.bin` + `.soc-version`**；`log_ca/` 等中间目录会被清理。
4. **record/report 直接使用 CANN 自带 cannsim**：与 camodel 同版本配套；CANN 9.1.0-beta.3 之前自带的 cannsim instr.bin 记录格式不兼容（424→432 字节），report 会报 `No executed instructions detected`。
5. **`output_bytes` 必须为 list 而非 tuple**：比对路径存在原地赋值（`outputs[idx] = ...`），回读侧已按 list 对齐。

### 环境要求

- Ascend950 仿真需安装对应 ops 包（如 `Ascend-cann-950-ops`）：Ascend950 的 TBE 算子实现与 aclnn 算子库（`libopapi*.so`）由它提供，缺失时 kernel 编译报 `OPTILING_FAILURE` / `*_OPERATOR_NOT_FOUND`，aclnn 执行报 `ACLNN_EXECUTE_FAILED`。
- kernel/aclnn 端到端（编译 → 仿真执行 → 回读 → 精度比对）与 `--sim-report` 流水图均已验证可用。
