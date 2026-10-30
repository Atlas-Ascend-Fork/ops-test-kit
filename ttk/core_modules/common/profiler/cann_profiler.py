#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""tf / onnx × NPU 共享 profiler：算子执行不经 torch dispatcher，下潜 CANN MsProfiler。

tf 算子的 GE 图执行与 aclmdl 的 kernel 对 torch dispatcher 均不可见，
且 torch_npu profiler 不得进入这两个 worker（test_structure 隔离纪律），
故下潜内部包 MsProfiler（aclprof 框架无关采集，ge/aclnn/kernel 同源），
解析 op_statistic 出 op 级 kernel 明细。

机制不可用（libmsprofiler.so 缺失 / aclprof init 失败 / 无 CSV 产出）
不降级墙钟，结果留空（Device/us 落 "----"），与 kernel/aclnn/geir 同口径。

采集窗对齐 e2e 循环协议（execute.py）：warmup 轮尾调 step() 开窗（warmup
轮不进窗）；warmup=0 时无 step() 可挂，__enter__ 即开窗。step() 幂等。
开窗前先同步设备：TF 设备执行异步，warmup kernel 尚在飞时 aclprofStart
会与其重叠导致失真或空采；onnx 的 synchronize 为 no-op。
"""

__all__ = ["CannMsProfiler"]

import contextlib
import logging
import os

from .base import FrameworkProfiler, ProfilerConfig, ProfileResult


class CannMsProfiler(FrameworkProfiler):
    """tf/onnx 共用的 msprof 采集器（aclprof 采集窗 + op_statistic 解析）。"""

    def __init__(self, config: ProfilerConfig, backend=None):
        self._testcase_name = config.testcase_name or "unknown"
        self._outdir = os.path.join(config.root_path, "msprof", "e2e", self._testcase_name)
        self._dev_id = config.dev_id
        self._warmup_count = config.warmup_count
        self._backend = backend
        self._prof = None
        self._prof_started = False

    def __enter__(self):
        if self._warmup_count <= 0:
            self._start_msprof()
        return self

    def __exit__(self, *exc):
        if self._prof is not None:
            self._prof.__exit__(*exc)

    def step(self):
        # warmup 轮尾：开启 aclprof 采集窗（幂等）
        self._start_msprof()

    def _start_msprof(self):
        if self._prof_started:
            return
        self._prof_started = True
        os.makedirs(self._outdir, exist_ok=True)
        # 开窗前等设备闲：TF 异步执行下 warmup kernel 可能仍在飞（onnx 为 no-op）
        if self._backend is not None:
            with contextlib.suppress(Exception):
                self._backend.synchronize(self._dev_id)
        # 懒 import：延迟 CANN 依赖，便于测试 monkeypatch
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
            # 机制不可用：采集留空（与 kernel/aclnn/geir 同口径），
            # 已初始化的 aclprof 会话在此清理
            if prof is not None:
                with contextlib.suppress(Exception):
                    prof.__exit__(None, None, None)
            self._prof = None
            logging.exception("CANN msprof init failed; perf columns left empty")
        else:
            self._prof = prof

    def result(self, repeat_count) -> ProfileResult:
        """op_statistic 聚合经 MsProfiler.result（同 ms_profiler 域共享解析）；
        机制不可用/无数据 → 空结果（不降级墙钟）。"""
        if self._prof is not None:
            return self._prof.result(repeat_count)
        return ProfileResult(source="disabled")
