# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""golden-mode NPU 共享执行器（kernel/aclnn/geir/e2e 四通路共用）。

npu_golden_runner：在 CANN msprof 采集窗内执行 golden，调用方注入无参闭包
（构造 NPU 张量 + 调用 golden + 返回通路侧 numpy 输出），返回
(outputs, ProfileResult, source)。NPU 执行/采集异常回落 CPU 闭包
（perf 空、source="cpu"），正确性优先。

- MsProfiler（libmsprofiler.so，API 视图 → op_statistic CSV）：任务级、
  框架无关，torch 算子与 tf.function 的 GE 任务同会话采集；
- start_step=warmup_count 隔离首轮；warmup 跟随 --warmup（与 DUT 同口径，默认 1）；
- task_profiling=False → 裸执行 + DisabledProfiler（列 "----"）；
- 设备初始化由调用方闭包负责（torch 闭包内 set_device / e2e 经 backend）；
- 张量 → numpy 转换由调用方闭包负责（各通路 dtype 语义不同）。
"""

import json
import logging
import os

from ..profiler.base import DisabledProfiler, ProfileResult
from ..profiler.ms_profiler import MsProfiler, TtkMsProfType

# golden 采集目录 = 通路 DUT 的 msprof 子目录 + 用例名 _golden 后缀
# （switches.test_mode → DUT 子目录：kernel=op / aclnn=op_api / geir / e2e）
_MSPROF_MODE_DIR = {"op": "op", "aclnn": "op_api", "geir": "geir", "framework-api": "e2e"}

_npu_device_capable = None


def has_npu_device() -> bool:
    """torch_npu 可导入且 device_count>0（真机与 camodel 仿真均满足）。

    进程内缓存一次（主进程预检探测，forkserver 子进程继承）；探测异常记 False，不抛出。
    """
    global _npu_device_capable
    if _npu_device_capable is None:
        try:
            import torch_npu

            _npu_device_capable = torch_npu.npu.device_count() > 0
        except Exception:
            _npu_device_capable = False
    return _npu_device_capable


_GOLDEN_SKIP_SENTINEL = "SUPPRESSED"
_PLACEHOLDER_CAUSE_CAP = 200


def npu_golden_runner(invoke_npu, invoke_cpu, dev_id, switches, testcase_name, warmup_count=None):
    """msprof 采集窗执行 NPU golden；失败回落 CPU。返回 (outputs, perf, source)。

    warmup_count=None 时跟随 --warmup（`1 if switches.warmup else 0`，
    与 DUT `_execute_eager` 同口径）；task_profiling=False 不开采集窗。

    回落与 task_profiling 开关无关：NPU 执行/采集失败 → CPU 闭包兜底
    （perf 空、source="cpu"）；CPU 闭包失败原样上抛（Invoke.golden 收敛为
    占位）；CPU 产出全为 FAIL 型占位串（UNSUPPORTED/GOLDEN_FAILURE 等）时
    把 NPU 根因追加进占位串，比对层按 str 哨兵透传到精度列。
    """
    profiling_enabled = bool(getattr(switches, "task_profiling", True))
    if warmup_count is None:
        warmup_count = 1 if getattr(switches, "warmup", True) else 0
    try:
        if not profiling_enabled:
            # 采集关闭：裸执行一次（无预热），失败同样走统一回落。
            return invoke_npu(), DisabledProfiler().result(1), "npu"
        # 目录随通路 DUT 的 msprof 子目录，用例名加 _golden 后缀与 DUT 数据
        # 同层：kernel → msprof/op/<case>_golden，e2e → msprof/e2e/<case>_golden
        mode = getattr(switches, "test_mode", None)
        mode_dir = _MSPROF_MODE_DIR.get(mode, mode or "unknown")
        outdir = os.path.join(switches.root_path, "msprof", mode_dir, f"{testcase_name}_golden")
        os.makedirs(outdir, exist_ok=True)
        prof = MsProfiler(dev_id, result_path=outdir, prof_type=TtkMsProfType.API, start_step=warmup_count)
        outputs = None
        prof.__enter__()
        try:
            for _i in range(warmup_count + 1):
                # start_step=warmup_count：前 warmup_count 个 step 不启动会话（预热隔离）
                prof.step()
                outputs = invoke_npu()
        finally:
            prof.__exit__(None, None, None)
        # 空数据（无 op_statistic）→ elapsed=0 → 结果列 "----"
        return outputs, prof.result(1), "npu"
    except Exception as npu_error:
        # NPU 执行/采集失败：CPU 闭包兜底，perf 空（source="cpu"）。
        logging.exception(
            "[%s] NPU golden execution failed (device error or msprof issue); "
            "falling back to CPU golden, device perf unavailable",
            testcase_name,
        )
        try:
            outputs = invoke_cpu()
        except Exception:
            logging.error(
                "[%s] CPU fallback golden also failed after NPU golden failure (%s: %s)",
                testcase_name,
                type(npu_error).__name__,
                npu_error,
            )
            raise
        return _annotate_npu_failure(outputs, testcase_name, npu_error), DisabledProfiler().result(1), "cpu"


def _annotate_npu_failure(outputs, testcase_name, npu_error):
    """CPU 回落产出全为 FAIL 型占位串时把 NPU 根因追加进占位串。

    条目含真实数组 / None / SUPPRESSED（PASS-SKIP 哨兵）即保持原样；
    全为占位串（NPU-only 算子 CPU 无真值）时逐条追加（比对层按 str 哨兵透传）。
    """
    entries = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
    if not entries or any(x is None or not isinstance(x, str) or x == _GOLDEN_SKIP_SENTINEL for x in entries):
        return outputs
    cause = f"{type(npu_error).__name__}: {npu_error}"
    if len(cause) > _PLACEHOLDER_CAUSE_CAP:
        cause = cause[:_PLACEHOLDER_CAUSE_CAP] + "..."
    logging.error(
        "[%s] golden unavailable on NPU and CPU fallback produced only placeholders; NPU failure cause: %s",
        testcase_name,
        cause,
    )
    annotated = [f"{x} (NPU golden failed: {cause})" for x in entries]
    return type(outputs)(annotated) if isinstance(outputs, (list, tuple)) else annotated[0]


def xpu_golden_perf(xpu_results, priority=None):
    """golden-mode XPU 三方执行的设备时间 → ProfileResult（source="xpu"）。

    xpu_results: {provider: {..., "perf": {"device_us": float|"NA"}}}；
    priority 优先命中，未命中取首个 device_us>0 条目；无效数据 → None
    （manual 回放无 perf 位 → None → 列 "----"）。
    """
    if not xpu_results:
        return None
    entries = []
    if priority is not None and priority in xpu_results:
        entries.append(xpu_results[priority])
    entries.extend(entry for provider, entry in xpu_results.items() if provider != priority)
    for entry in entries:
        perf = (entry or {}).get("perf") or {}
        try:
            device_us = float(perf.get("device_us"))
        except (TypeError, ValueError):
            continue
        if device_us > 0:
            return ProfileResult(elapsed_us=device_us, source="xpu")
    return None


def golden_perf_columns(perf, source):
    """ProfileResult → golden 性能四列 dict（device/count/details/source）。

    口径与 e2e FrameworkApiReturnStructure.set_golden_perf 一致：
    elapsed<=0 → "----"；kernel_details 空不写 count/details。
    """
    cols = {
        "golden_device_perf_us": None,
        "golden_kernel_count": None,
        "golden_kernel_details": None,
        "golden_source": source,
    }
    if perf is None or perf.elapsed_us <= 0:
        cols["golden_device_perf_us"] = "----"
        return cols
    cols["golden_device_perf_us"] = f"{perf.elapsed_us:.3f}"
    kd = perf.kernel_details
    if kd and kd.kernels:
        cols["golden_kernel_count"] = str(len(kd.kernels))
        cols["golden_kernel_details"] = json.dumps(
            [
                {
                    "name": k.name,
                    "total": round(k.device_us, 3),
                    "avg": round(k.avg_us, 3),
                    "max": round(k.max_us, 3),
                    "min": round(k.min_us, 3),
                    "calls": k.calls,
                }
                for k in kd.kernels
            ],
            ensure_ascii=False,
        )
    return cols
