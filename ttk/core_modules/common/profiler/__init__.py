#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""common/profiler：性能采集与解析。

采集与解析两层：
- **框架层 profiler**（torch_profiler.py / cann_profiler.py）以 context manager 包裹被测执行，
  产出统一 ProfileResult（elapsed_us + kernel_details）——解析自己负责
  （kernel_details.csv / operator_details.csv / 事件表各家自己读），通路与
  ReturnStructure 只消费统一结果。
- **CANN 采集器 MsProfiler**（ms_profiler.py，libmsprofiler.so ctypes +
  msprof.py 子进程导出 summary/timeline）只负责采集与落盘；Kernel/ACLNN/GEIR
  通路在采集点直接构造（result_path/prof_type/start_step 内嵌于 launch
  循环编排），落盘 CSV 由各通路解析（kernel/execute.py 的 op_summary、
  aclnn/execute.py 的 op_statistic、geir/perf_parse.py）。

文件布局（框架侧按"框架族"切分——factory 按 api_name 前缀在同族
互斥分支中二选一，文件与路由心智一致）：
- base.py：契约层——Profiler ABC（= FrameworkProfiler）+ KernelInfo/
  KernelDetails/ProfileResult/ProfilerConfig 数据类 + _self_device_time
  共享内核 + Disabled/WallClock 平凡默认实现
- torch_profiler.py：torch 系——TorchProfiler（torch.profiler）+
  TorchNpuProfiler（torch_npu.profiler）+ resolve_activities/summarize_events
  共享引擎（E2E 与 server torch pass 同源消费）
- cann_profiler.py：tf/onnx 系 × NPU 共享——CannMsProfiler（tf 算子与
  aclmdl 执行均不经 torch dispatcher，下潜 CANN 层出 op 级 kernel 明细；
  机制不可用结果留空，与 kernel/aclnn/geir 同口径）
- ms_profiler.py：MsProfiler（见上）
- types.py：ACL 错误码表 + AclProfType/TtkMsProfType/AiCoreProfMetrics 枚举
  + msprof ctypes 结构（MsprofCompactInfo 族）
- factory.py：get_profiler——E2E 按 api_name 前缀 + backend 分派
  （XPU pass 见 ttk/remote/server/xpu_profiler.py——唯一消费方为 server，
  torch 门面复用本包 torch_profiler 共享引擎）
"""

__all__ = [
    "Profiler",
    "ProfilerConfig",
    "ProfileResult",
    "KernelDetails",
    "KernelInfo",
    "get_profiler",
    "MsProfiler",
    "MsProfOpDfx",
    "TtkMsProfType",
    "AclProfType",
]

from .base import KernelDetails, KernelInfo, Profiler, ProfilerConfig, ProfileResult
from .factory import get_profiler
from .ms_profiler import MsProfiler
from .types import AclProfType, MsProfOpDfx, TtkMsProfType
