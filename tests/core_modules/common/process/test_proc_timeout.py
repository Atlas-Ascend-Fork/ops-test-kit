# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""--proc-timeout 判据单测：状态门控 + stage 豁免白名单。

回归点：旧子串判据（Profiling/Compilation/Gen/Waiting）漏了 tiling、比对、
EstablishInterface 等阶段——这些 stage 挂死不受超时保护。"""

import time
from types import SimpleNamespace

import pytest

from ttk.core_modules.common.process.simple_command import (
    PROCESS_STATUS_CODE,
    SimpleCommandProcess,
)


def _proc_at(status, stage, timeout=10, age=100):
    """构造不 spawn 真进程的 SimpleCommandProcess 桩（仅 _update 依赖面）。"""
    proc = SimpleCommandProcess.__new__(SimpleCommandProcess)
    proc.status = status
    proc.timeout = timeout
    proc.data = {"stage": stage} if stage else {}
    proc.process_status_timestamp = time.time() - age
    proc.parent_pipe = SimpleNamespace(poll=lambda: False)
    proc.parent = SimpleNamespace(exitcode=None)
    return proc


class TestTimeoutRule:
    def test_tiling_stage_hang_now_covered(self):
        proc = _proc_at(PROCESS_STATUS_CODE.RUNNING, "OnDynOpTiling")
        with pytest.raises(TimeoutError):
            proc._update()

    def test_comparison_and_establish_interface_covered(self):
        for stage in ("OnComparison", "DynEstablishInterface", "UNKNOWN"):
            with pytest.raises(TimeoutError):
                _proc_at(PROCESS_STATUS_CODE.RUNNING, stage)._update()

    def test_lock_wait_exempt(self):
        proc = _proc_at(PROCESS_STATUS_CODE.RUNNING, "OnAcquireLock")
        proc._update()  # 协调性等待不受罚

    def test_idle_worker_exempt_even_with_stale_stage(self):
        proc = _proc_at(PROCESS_STATUS_CODE.IDLE, "OnProfilingPrint")
        proc._update()

    def test_bootstrap_stuck_covered(self):
        proc = _proc_at(PROCESS_STATUS_CODE.LAUNCHED, None)  # LAUNCHED 后卡 bootstrap
        with pytest.raises(TimeoutError):
            proc._update()

    def test_within_timeout_no_fire(self):
        proc = _proc_at(PROCESS_STATUS_CODE.RUNNING, "OnDynOpTiling", age=1)
        proc._update()

    def test_xpu_collect_wait_exempt(self):
        """OnXpuCollect（XPU/NPU 重叠的远端收割等待）豁免——协调性等待语义，
        join 上界由 dispatch deadline 兜底，不受 --proc-timeout 惩罚。"""
        proc = _proc_at(PROCESS_STATUS_CODE.RUNNING, "OnXpuCollect")
        proc._update()  # 远端等待不受罚

    def test_timeout_disabled_never_fires(self):
        proc = _proc_at(PROCESS_STATUS_CODE.RUNNING, "OnDynOpTiling", timeout=0)
        proc._update()
