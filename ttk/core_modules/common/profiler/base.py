#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""common/profiler 契约层。

Profiler ABC（context manager + result()）与统一结果数据类，以及两个
平凡默认实现（Disabled / WallClock）。_self_device_time 是 torch 系
profiler 与 server XPU pass 共享的设备时间提取内核（单一实现防漂移）。

warmup/repeat 由调用方控制——profiler 只在 with 块内采集。
框架侧实现见 torch_profiler.py（torch 系）与 cann_profiler.py（tf/onnx 系）；
CANN 采集器见 ms_profiler.py；server XPU 门面见 ttk/remote/server/xpu_profiler.py。
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class KernelInfo:
    name: str
    device_us: float
    calls: int
    avg_us: float
    max_us: float = 0.0
    min_us: float = 0.0


@dataclass
class KernelDetails:
    kernels: List[KernelInfo] = field(default_factory=list)
    total_device_us: float = 0.0
    total_cpu_us: float = 0.0


@dataclass
class ProfileResult:
    """统一结果。elapsed_us 为**单轮口径**（采集窗内设备忙碌时间 / repeat，
    或 op 级 Σ Avg——Avg 已被 msprof 按实际采集次数平均）。

    wall_us：侧路 wall-clock（TfNpu 降级归因用），无采集为 None。
    source：归因标识——torch_npu / msprof / torch / wallclock / disabled。
    """

    elapsed_us: float = 0.0
    kernel_details: Optional[KernelDetails] = None
    wall_us: Optional[float] = None
    source: str = ""


@dataclass
class ProfilerConfig:
    """Runtime options shared by the framework profiler implementations."""

    testcase_name: str = ""
    root_path: str = "."
    dev_id: int = 0
    enabled: bool = True
    warmup_count: int = 0
    aclgraph: bool = False
    graph: bool = False


def _self_device_time(evt, device: str, device_time_attr: Optional[str] = None) -> float:
    """3-candidate self device-time extraction for one profiler event (pure self_).

    Order: explicit device_time_attr (if configured) → self_device_time_total
    (torch 2.7+ cuda/musa unified) → self_{device}_time_total (legacy,
    e.g. self_cuda_time_total / self_mlu_time_total).

    Uses ``is not None`` (not truthiness) so a legitimate v=0.0 idle kernel
    is not mis-treated as missing and falls through to a wrong fallback.

    Only ``self_*`` fields are used (self time, excluding nested child ops);
    ``device_time_total`` (total time, includes nested) was dropped to avoid
    mixing self_/total semantics — it would inflate the sum when nested ops
    double-count, and it was redundant (self_device_time_total already
    covers the musa scenario it was meant for).

    Shared by torch_profiler（E2E TorchProfiler + summarize_events 引擎）
    and server 的 torch pass（ttk/remote/server/xpu_profiler）——单一实现防漂移。
    """
    if device_time_attr is not None:
        return getattr(evt, device_time_attr, 0.0)
    v = getattr(evt, "self_device_time_total", None)
    if v is not None:
        return v
    return getattr(evt, f"self_{device}_time_total", 0.0)


class FrameworkProfiler(ABC):
    """Framework-level profiler abstraction (context manager)."""

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, *exc):
        pass

    @abstractmethod
    def result(self, repeat_count) -> ProfileResult:
        pass

    def step(self):  # noqa: B027
        pass


# 契约名（Profiler 即 ABC，通路消费经此名）
Profiler = FrameworkProfiler


class DisabledProfiler(FrameworkProfiler):
    """No-op profiler used when task profiling is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def result(self, repeat_count) -> ProfileResult:
        return ProfileResult(source="disabled")


class WallClockProfiler(FrameworkProfiler):
    """兜底墙钟 profiler（非 NPU waive / 未知框架）。

    step() 重置计时窗：e2e 循环在 warmup 轮尾调用（窗口自首个计时轮起）；
    不调 step() 时窗口自 __enter__ 起。
    """

    def __init__(self, backend):
        self._start = None
        self._elapsed = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._elapsed = time.perf_counter() - self._start

    def step(self):
        self._start = time.perf_counter()

    def result(self, repeat_count) -> ProfileResult:
        elapsed_us = (self._elapsed or 0.0) * 1e6 / max(repeat_count, 1)
        return ProfileResult(
            elapsed_us=elapsed_us,
            kernel_details=None,
            wall_us=elapsed_us,
            source="wallclock",
        )
