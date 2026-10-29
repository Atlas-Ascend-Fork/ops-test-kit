# NPUSim 仿真排障（--backend npusim）

`--backend npusim` 走 SoC 级仿真（CANN 自带 cannsim + camodel），报错与真机不同，需单独排查。

**先决条件**：须 source 含 **Ascend-950-ops** 的 CANN 环境（如 `9.1.0-beta.3`）。CANN **9.2.0 缺 950-ops**，算子编译/执行会在早期失败。

## 常见报错

| 报错 / 现象 | 原因 | 排查 / 修复 |
|---|---|---|
| `OPTILING_FAILURE` / `*_OPERATOR_NOT_FOUND` | 缺 950-ops 包或该算子 TBE 实现 | 确认已装 `Ascend-cann-950-ops`；看 `ttk-debug.log` |
| `ACLNN_EXECUTE_FAILED` | aclnn 算子库不支持该 SoC | 同左，确认 950-ops 已装 |
| `SIM_RESULT_MISSING` / `wrapper_error.json` 有内容 | wrapper 执行异常 | 看 `<sim_output>/<case>/wrapper_error.json` 与 `record_out/cannsim_*/cannsim.log` |
| `RuntimeError: Current device only support aclnn operator` / `ERR00007 PTA feature not supported` | e2e 调用了非 aclnn 的 legacy 自定义算子（如 `torch_npu.npu_conv2d`），camodel 仅支持 aclnn 算子 | 换用有 aclnn 实现的算子，或该算子改用真机 `--backend npu` |
| `precision_status: FAIL` | golden 与仿真输出偏差 | 检查容差、golden 实现、shape/format 定义 |
| report 报 `No executed instructions` | 用了格式不匹配的 cannsim 版本 | 确认 `--sim-report` 走环境自带 `$ASCEND_TOOLKIT_HOME/bin/cannsim` |
| `CANN cannsim not found` | 未安装含 cannsim 的 CANN，或 `cannsim` 不在 PATH | 确认 `$ASCEND_TOOLKIT_HOME/bin/cannsim` 存在 |
| 用例卡死：`OnDynProfiling` 长时间 RUNNING，`cannsim.log` 停在 `TASK_BEGIN RTSQ_1` 无 `TASK_DONE` | Ascend950 camodel teardown（`rtCtxDestroy`）对个别用例忙自旋 | **已内置规避**：wrapper 用 `skip_teardown=True` 跳过 teardown，无需干预 |

## 已知限制（非报错，勿当 bug 排查）

- 仅支持 NPUSim 注册的 SoC：`Ascend950` / `Ascend950DT`。
- 仿真模式 `*_perf_us` 列为 `UNKNOWN`（model 模式无单值周期），性能分析看 `--sim-report` 的流水图。
- e2e 仿真仅支持 **aclnn 算子**：`torch.abs`、inplace relu 等在 camodel 下会挂起，先小 shape 单用例验证算子受支持。
- 仿真慢：camodel 初始化 20–30s，大 shape 用例分钟级起，建议 `--sim-cores 0` + 小 shape。
- 并发进程默认自动设为 1（避免多 camodel 打爆 CPU）。

完整排障与使用：`docs/npusim/npusim_usage_guide.md` §7
