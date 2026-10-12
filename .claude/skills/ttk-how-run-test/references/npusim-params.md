# NPUSim 仿真参数（--backend npusim）

kernel / aclnn / e2e 子命令均可加 `--backend npusim`，把真机执行替换为 NPUSim（CANN 自带 cannsim + camodel，SoC 级周期精确仿真）仿真执行。golden 生成、精度比对、结果 CSV 全部复用现有链路，无卡即可验证算子精度。

## 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--backend {npu,npusim}` | `npu` | `npusim` 走仿真执行（e2e 另支持 `cpu`；三方采集仅走远端，无 `xpu` 本机后端） |
| `--sim-soc <SoC>` | `Ascend950` | 仿真 SoC（`Ascend950` / `Ascend950DT`） |
| `--sim-output <dir>` | `<root>/sim_output` | 仿真中间产物根目录 |
| `--sim-report` | 关闭 | 仿真结束后生成流水图（`trace_core*.json` + HTML 报告） |
| `--sim-cores <范围>` | 全核 | 指定仿真核，如 `0-2,12-14` / `all` / `0`；建议小范围以加快仿真 |
| `--sim-obj <kernel.o>` | 无 | 传给 report 的目标文件（含源码/汇编级信息） |

## 约束

- 与 `--no-prof` **互斥**：仿真后端内部自动生成输入/golden，不能再走手工数据 prepare（`sim_args.apply_sim_args` 直接报错）。
- **GEIR 不支持**仿真。
- e2e 仿真聚焦 **eager** 执行；graph / aclgraph / fullgraph 依赖 GE 图编译（camodel 缺 pcie bar 能力），仿真下未验证、可能失败，建议仅用 eager。
- e2e 仿真要求算子**在线编译**：`-b`/`--binary`（默认二进制复用）被强制关闭并告警。
- 并发进程默认自动设为 1（避免多 camodel 实例打爆 CPU）。

## 环境要求

- 必须 source 含 **Ascend-950-ops** 包的 CANN 环境（如 `9.1.0-beta.3` + `Ascend-cann-950-ops`）。**9.2.0 缺 950-ops**，算子编译/执行会失败（`OPTILING_FAILURE` / `*_OPERATOR_NOT_FOUND` / `ACLNN_EXECUTE_FAILED`）。
- 直接使用 CANN 自带 `cannsim`（`$ASCEND_TOOLKIT_HOME/bin/cannsim`，与 camodel 同版本配套），**无需另装 npu-simulator 仓库源码**。

## 示例

```shell
# Kernel：仿真 + 精度比对（核 0，加快仿真）
python3 -m ttk kernel --backend npusim -i examples/case_store/kernel/cases.csv -t add_fp16_broadcast --sim-cores 0

# 筛选单用例 + 输出流水图
python3 -m ttk kernel --backend npusim -i cases.csv -t add_01 --sim-cores 0 --sim-report -o result.csv

# ACLNN
python3 -m ttk aclnn --backend npusim -i examples/case_store/aclnn/cases.csv -t add_fp16_broadcast --sim-cores 0

# E2E（eager）额外出流水图
python3 -m ttk e2e --backend npusim -i examples/case_store/e2e/torch_cases.csv -t add_f32_alpha --sim-report
```

## 产物

```
<sim_output>/<case_name>/
├── record_out/cannsim_<ts>_<case>/
│   ├── instr.bin      # 指令流（report 输入，归档根目录）
│   └── cannsim.log    # record 日志
└── report/            # 仅 --sim-report
    └── trace_core0.json  # ★ Chrome tracing 流水图
```

仿真模式 `*_perf_us` 列为 `UNKNOWN`（model 模式无单值周期），性能分析请查看流水图。

## 已知限制

- 仅支持 NPUSim 注册的 SoC（`Ascend950` / `Ascend950DT`）。
- 仿真较慢：camodel 初始化 20–30s，大 shape 用例分钟级起，建议 `--sim-cores 0` + 小 shape 先行验证。
- e2e 仿真仅支持 **aclnn 算子**：非 aclnn 的 legacy 自定义算子（如 `torch_npu.npu_conv2d`）会被 torch_npu 拒绝；部分算子（`torch.abs`、inplace relu）在 camodel 下会挂起，先小 shape 验证。

完整指南与排障：`docs/NPUSim/npusim_usage.md`
