#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You should refer to the License for details.
# You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.


import importlib.util
import logging

from .api_resolver import resolve_api
from .base import Backend
from .framework_detector import detect_framework, is_inplace_tensor_method

__all__ = ["Backend", "get_backend", "resolve_api", "detect_framework", "is_inplace_tensor_method"]

_log = logging.getLogger(__name__)


def _npu_torch_available() -> bool:
    """torch 侧 NPU 探测：import torch_npu 后 torch.npu.is_available()。

    ImportError（未安装）/ RuntimeError（设备不可用）等一律告警并回落 CPU。"""
    try:
        import torch
        import torch_npu  # noqa: F401  # torch.npu 属性随 torch_npu 注册

        return bool(torch.npu.is_available())
    except Exception as e:
        _log.warning("NPU probe failed, falling back to CPU: %s", e)
        return False


def _has_npu_device() -> bool:
    """tf 侧 NPU 探测：npu_device 包是否安装（open 由 backend.set_device 延迟触发）。"""
    try:
        return importlib.util.find_spec("npu_device") is not None
    except Exception:
        return False


def get_backend(force_cpu: bool = False, framework: str = "torch") -> Backend:
    """Resolve a hardware Backend: framework × (CPU | NPU)。

    - ``force_cpu``（--backend cpu，e2e）→ CPU 系
    - NPU：tf 看 npu_device 安装；torch 看 torch_npu + torch.npu.is_available()
    - 探测失败回落 CPU（告警）；实例不缓存，每次新建。

    三方标杆采集（--golden-mode XPU / --xpu-perf）走 remote endpoints
    （common/xpu.collect_third_party → ttk.remote.client），不经本函数。
    """
    if framework == "tf":
        if not force_cpu and _has_npu_device():
            from .npu_tf_backend import NpuTfBackend

            return NpuTfBackend()
        from .cpu_tf_backend import CpuTfBackend

        return CpuTfBackend()

    if not force_cpu and _npu_torch_available():
        from .npu_torch_backend import NpuTorchBackend

        return NpuTorchBackend()
    from .cpu_torch_backend import CpuTorchBackend

    return CpuTorchBackend()
