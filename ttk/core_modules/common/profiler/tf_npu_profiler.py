#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""tf 系 × NPU profiler。

TF 算子不走 torch dispatcher，torch_npu.profiler 看不见；tf.profiler 的
设备面（xplane device plane）不支持第三方 NPU 设备插件——NPU 内核对两者
都不可见。故下潜 CANN 层：内部包 MsProfiler（aclprof 框架无关采集，ge/
aclnn/kernel 同源），解析 op_statistic 统计；机制不可用降级 wall-clock
（"Eager mode falls back to wall-clock timing"）。
"""

import contextlib
import logging
import os

from .base import FrameworkProfiler, KernelDetails, ProfileResult, WallClockProfiler
from .ms_profiler import find_newest_csv, read_op_statistic


class TfNpuProfiler(FrameworkProfiler):
    """Profiler for TF NPU using CANN msprof directly.

    TF ops do not go through torch's dispatcher, so torch_npu.profiler's
    kernel_details.csv is not produced.  We use MsProfiler (libmsprofiler.so)
    to collect CANN task-level records, then parse op_statistic CSV for
    per-op device timing.  Eager mode falls back to wall-clock timing.
    """

    def __init__(self, backend, testcase_name="", root_path=".", dev_id=0):
        self._testcase_name = testcase_name or "unknown"
        self._outdir = os.path.join(root_path, "msprof", "e2e", self._testcase_name)
        os.makedirs(self._outdir, exist_ok=True)
        self._dev_id = dev_id
        self._prof = None
        self._wall = WallClockProfiler(backend)

    def __enter__(self):
        # 懒 import：既延迟 CANN 依赖，也让测试可 monkeypatch
        # ms_profiler.MsProfiler（模块属性在每次 enter 时重新解析）
        from .ms_profiler import MsProfiler, TtkMsProfType

        prof = None
        try:
            prof = MsProfiler(
                self._dev_id,
                result_path=self._outdir,
                prof_type=TtkMsProfType.API,
                start_step=0,
            )
            prof.__enter__()
            prof.step()
        except Exception:
            # 机制不可用（libmsprofiler.so 缺失 / aclprof init 中途失败）→ 降级
            # wall-clock，兑现类语义 "Eager mode falls back to wall-clock timing"。
            # 部分初始化的 aclprof 会话经 __exit__ 守卫清理，避免泄漏。
            if prof is not None:
                with contextlib.suppress(Exception):
                    prof.__exit__(None, None, None)
            self._prof = None
            logging.exception("TF NPU profiler (CANN msprof) init failed; falling back to wall-clock")
        else:
            self._prof = prof
        self._wall.__enter__()
        return self

    def __exit__(self, *exc):
        self._wall.__exit__(*exc)
        if self._prof:
            self._prof.__exit__(*exc)

    def result(self, repeat_count) -> ProfileResult:
        """msprof op_statistic 聚合（解析归 ms_profiler 域的共享 reader，
        谁采集谁解析）；无数据/解析失败降级 wall-clock。wall 侧路
        始终采集，msprof 命中时经 wall_us 透出（降级归因）。"""
        wall_res = self._wall.result(repeat_count)
        csv_path = find_newest_csv(self._outdir, ("op_statistic_",))
        if csv_path:
            kernels = []
            try:
                kernels = read_op_statistic(csv_path)
                total_avg = round(sum(k.avg_us for k in kernels), 3)
            except Exception as e:
                logging.warning(f"Failed to parse {csv_path}: {e}")
                total_avg = 0.0
            if total_avg > 0:
                return ProfileResult(
                    elapsed_us=total_avg,
                    kernel_details=KernelDetails(
                        kernels=kernels,
                        total_device_us=total_avg * max(repeat_count, 1),
                        total_cpu_us=0.0,
                    ),
                    wall_us=wall_res.elapsed_us,
                    source="msprof",
                )
        return wall_res
