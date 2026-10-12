#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host-memory gate before per-case heavy workloads (worker side).

Snapshot is cgroup-aware: in limited containers the host free figure is a
mirage (e.g. 754 GB host / 32 GB cgroup), so headroom is computed from
min(cgroup limit, host total).

Concurrent worker spawns can be OOM-killed outright when free memory is
exhausted; the gate pauses the workload at critically low headroom
instead. The policy is deliberately lenient:

- trips only at critically low headroom: ``available < min(floor, total*ratio)``
  (large machines hit the absolute floor — big legitimate workloads are
  never blocked; small machines scale by ratio)
- waits at most ``timeout_s`` seconds, then proceeds with a warning
  (never blocks forever)
- ``/tmp/TTK_FORCE_MEMORY_OVERRIDE`` (legacy ops escape hatch) releases
  the gate immediately
- ``--no-memory-check`` skips the gate entirely
"""

import logging
import os
import time

from ....utilities.memory import cgroup_mem_current, cgroup_mem_limit

__all__ = ["gate_host_memory", "host_memory_floor"]

OVERRIDE_FILE = "/tmp/TTK_FORCE_MEMORY_OVERRIDE"
_GB = 1024**3


def _memory_snapshot():
    """(total, available)：有限额容器按 cgroup 算（宿主 free 是假象）。

    available 取 cgroup 限额余量与宿主可用的较小者——两者任一耗尽都会
    OOM kill；无 cgroup（裸机）退化为宿主快照。
    """
    import psutil

    vm = psutil.virtual_memory()
    limit = cgroup_mem_limit()
    if limit is None:
        return vm.total, vm.available
    total = min(limit, vm.total)
    available = max(0, limit - (cgroup_mem_current() or 0))
    return total, min(available, vm.available)


def host_memory_floor(total_bytes: int, *, floor_gb: float = 8.0, ratio: float = 0.1) -> float:
    """Critical watermark = min(floor_gb, total * ratio)——大机器取绝对下限，小机器按比例。"""
    return min(floor_gb * _GB, total_bytes * ratio)


def gate_host_memory(no_check: bool = False, process_ctx=None, *, timeout_s: float = 300.0, poll_s: float = 1.0):
    """拉起重负载（输入/golden/设备执行）前的 host 内存闸门（宽松版）。

    Args:
        no_check: switches.no_memory_check（--no-memory-check）→ 直接返回
        process_ctx: worker 进程上下文（进入等待时上报 OnWaitingForMemory）
        timeout_s: 低于水位时的最长等待（超时放行并告警，不死等）
        poll_s: 轮询间隔
    """
    if no_check:
        return

    total, available = _memory_snapshot()
    floor = host_memory_floor(total)
    if available >= floor:
        return
    if process_ctx is not None:
        process_ctx.notify_status("OnWaitingForMemory")
    logging.warning(
        "Memory low: available %.1f GB < floor %.1f GB (limit %.1f GB); waiting up to %.0f s (override: touch %s)",
        available / _GB,
        floor / _GB,
        total / _GB,
        timeout_s,
        OVERRIDE_FILE,
    )
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        time.sleep(poll_s)
        if os.path.exists(OVERRIDE_FILE):
            logging.warning("Memory gate override detected (%s), continuing", OVERRIDE_FILE)
            return
        total, available = _memory_snapshot()
        if available >= floor:
            logging.info("Memory recovered to %.1f GB, continuing", available / _GB)
            return
    logging.warning(
        "Memory still low after %.0f s (available %.1f GB); proceeding anyway",
        timeout_s,
        _memory_snapshot()[1] / _GB,
    )
