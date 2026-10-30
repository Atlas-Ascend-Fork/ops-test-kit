#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""get_profiler 工厂路由。

E2E（framework-api）通路的 profiler 分派，按 api_name 前缀 + backend 路由：
  task_prof == False          → DisabledProfiler
  tf.*/tensorflow.* + NPU     → CannMsProfiler（算子不经 torch dispatcher，
                               下潜内部包 MsProfiler 出 op 级 kernel 明细；
                               机制不可用结果留空——与 kernel/aclnn/geir
                               同口径，不降级墙钟）
  tf.*/tensorflow.* + 其他    → WallClockProfiler
  onnx.* + NPU                → CannMsProfiler（同上，aclmdl 执行）
  onnx.* + 其他               → WallClockProfiler（CPU 侧 waive 墙钟）
  torch_npu.* + 非 NPU        → raise（torch_npu API 必须在 NPU 上跑）
  torch.* + NPU builtin       → TorchNpuProfiler（torch_npu.profiler）
  torch.* + 其他              → TorchProfiler
  其余 + NPU                  → TorchNpuProfiler；其余 → WallClockProfiler

Kernel/ACLNN/GEIR 通路不经过本工厂：MsProfiler 构造依赖通路侧的
result_path/prof_type/start_step 等编排数据，在各自采集点直接构造
（kernel/execute.py、aclnn/execute.py、geir/worker.py）。四通路
统一入口。
"""

__all__ = ["get_profiler"]

from typing import Optional

from .base import DisabledProfiler, ProfilerConfig, WallClockProfiler
from .torch_profiler import TorchNpuProfiler, TorchProfiler


def get_profiler(
    api_name: str,
    backend,
    config: Optional[ProfilerConfig] = None,
):
    """E2E 工厂：按 api_name 前缀 + backend 分派。config 中的
    testcase_name/root_path/dev_id/warmup_count/aclgraph 由工厂解包传给
    选中实现，其余字段由实现自决。"""
    if config is None:
        config = ProfilerConfig()
    elif not isinstance(config, ProfilerConfig):
        raise TypeError(f"config must be ProfilerConfig, got {type(config).__name__}")

    if not config.enabled:
        return DisabledProfiler()

    # E2E 框架分派（按 api_name 前缀 + backend）
    if api_name.startswith(("tf.", "tensorflow.", "onnx.")):
        # tf 算子与 aclmdl 执行均不经 torch dispatcher，且 torch_npu profiler
        # 不得进入这两个 worker（test_structure 隔离纪律）：下潜 CANN
        # MsProfiler；机制不可用结果留空。非 NPU 无设备执行，墙钟 waive。
        if backend.is_npu():
            from .cann_profiler import CannMsProfiler

            return CannMsProfiler(config, backend)
        return WallClockProfiler(backend)
    if api_name.startswith("torch_npu."):
        if not backend.is_npu():
            raise RuntimeError(f"API '{api_name}' requires NPU backend, but current is '{backend.device_type()}'")
        return TorchNpuProfiler(backend, config)
    if api_name.startswith("torch."):
        if backend.is_npu() and backend.profile.get("profiler") == "builtin":
            return TorchNpuProfiler(backend, config)
        return TorchProfiler(backend)

    if backend.is_npu():
        return TorchNpuProfiler(backend, config)
    return WallClockProfiler(backend)
