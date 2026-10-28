#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You should refer to the License for details.
# You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.


import importlib
import importlib.util
import logging
from typing import Optional

from ttk.config.loader import get_hardware_config

from .api_resolver import resolve_api
from .base import Backend
from .framework_detector import detect_framework, is_inplace_tensor_method

__all__ = ["Backend", "get_backend", "resolve_api", "detect_framework", "is_inplace_tensor_method"]

_log = logging.getLogger(__name__)

# Module-level cache of the frameworks config segment (lazy-loaded once).
_HW_PROFILES_CACHE: Optional[dict] = None


def _hw_profiles(framework: str) -> dict:
    """Return the hardware-profile dict for a framework (cached).

    Lazily loads the ``frameworks`` config segment on first call and caches it
    module-level. Returns the per-framework sub-dict (role -> profile); empty
    dict when the framework is absent.
    """
    global _HW_PROFILES_CACHE
    if _HW_PROFILES_CACHE is None:
        _HW_PROFILES_CACHE = get_hardware_config()
    return _HW_PROFILES_CACHE.get(framework, {})


def _validate_profile(name: str, profile: dict) -> None:
    """Fail-fast validation of a hardware profile dict.

    - torch_lib must be present (torch module attribute name).
    - profiler must be the literal 'builtin' or a dict carrying 'activities'.
    - 'builtin' is NPU-only:非 NPU profile 会路由到 TorchProfiler，其按
      activities 消费配置，builtin 会在运行时 TypeError。
    """
    if "torch_lib" not in profile:
        raise ValueError(f"profile '{name}' missing torch_lib")
    prof = profile.get("profiler")
    if prof == "builtin" and profile["torch_lib"] != "npu":
        raise ValueError(f"profile '{name}' profiler 'builtin' only supports torch_lib 'npu'")
    if prof != "builtin" and not (isinstance(prof, dict) and "activities" in prof):
        raise ValueError(f"profile '{name}' profiler must be 'builtin' or dict with activities")


def _build(framework: str, name: str, profile: dict) -> Backend:
    """Validate the profile then instantiate and inject torch_lib/profile/segment.

    The backend class is derived from ``torch_lib`` (the torch module attribute
    name), NOT from the segment name: ``npu`` -> NpuTorchBackend, ``cpu`` ->
    CpuTorchBackend, anything else (cuda/mlu/musa/...) -> XpuTorchBackend. The
    segment name (the yaml key, e.g. 'xpu') is injected as ``_segment_name`` so
    alias() returns whatever the deployer named the segment.

    Fail-fast: invalid profiles raise before any backend is created.
    """
    _validate_profile(name, profile)
    torch_lib = profile["torch_lib"]
    if torch_lib == "npu":
        from .npu_torch_backend import NpuTorchBackend

        cls = NpuTorchBackend
    elif torch_lib == "cpu":
        from .cpu_torch_backend import CpuTorchBackend

        cls = CpuTorchBackend
    else:  # cuda / mlu / musa / other -> generic accelerator
        from .xpu_torch_backend import XpuTorchBackend

        cls = XpuTorchBackend
    b = cls()
    b.torch_lib = torch_lib
    b.profile = profile
    b._segment_name = name  # alias = segment name (arbitrary)
    return b


def _probe(profile: dict) -> bool:
    """Probe whether a hardware profile's torch_lib is usable.

    cuda is native to torch (no ``torch_cuda`` module to import); other libs
    (npu/mlu/...) require importing ``torch_<lib>`` first. The whole body
    is wrapped so ImportError (lib not installed), AttributeError/None (torch
    attr missing), and RuntimeError (e.g. PrivateUse1 single-slot already
    renamed) all degrade to a logged warning + False.

    A missing/empty ``torch_lib`` is checked up front (distinct warning) so a
    malformed config is not silently swallowed as "CPU fallback".
    """
    lib = profile.get("torch_lib")
    if not lib:
        _log.warning("hardware probe skipped: profile missing torch_lib (%s)", profile)
        return False
    try:
        import torch

        if lib != "cuda":
            importlib.import_module(f"torch_{lib}")
        mod = getattr(torch, lib, None)
        return bool(mod and mod.is_available())
    except Exception as e:
        _log.warning(
            "hardware probe failed for torch_lib=%s: %s",
            lib,
            e,
        )
        return False


def _has_npu_device() -> bool:
    """tf 侧 NPU 探测：npu_device 包是否安装（open 由 backend.set_device 延迟触发）。"""
    try:
        return importlib.util.find_spec("npu_device") is not None
    except Exception:
        return False


def _npu_onnx_available() -> bool:
    """onnx 侧 NPU 探测：DSMI 计数 > 0。

    **不加载 libascendcl**——so 加载属通路层职责（framework_api/onnx/runtime），
    get_backend 阶段（父进程/探测期）零 ACL 触碰。DSMI 缺失/异常告警回落 CPU。
    """
    try:
        from ..device.dsmi.interface import DSMIInterface

        return DSMIInterface().get_device_count() > 0
    except Exception as e:
        _log.warning("onnx NPU probe failed, falling back to CPU: %s", e)
        return False


def get_backend(force_cpu: bool = False, framework: str = "torch") -> Backend:
    """Resolve a hardware Backend.

    - ``force_cpu``（--backend cpu，e2e）→ CPU 系
    - torch：iterate ``_hw_profiles("torch")`` in declared order, skip profiles
      whose ``torch_lib`` is 'cpu' (CPU is the fallback below), ``_probe`` each;
      first hit is ``_build``. Nothing detected -> CpuTorchBackend fallback.
    - tf：npu_device 包安装探测，失败回落 CPU（告警）
    - onnx：DSMI 设备计数探测（不触 libascendcl），失败回落 CPU（告警）

    实例不缓存，每次新建。三方标杆采集（--golden-mode XPU / --xpu-perf）走
    remote endpoints（common/xpu.collect_third_party → ttk.remote.client），
    不经本函数。
    """
    if framework == "tf":
        if not force_cpu and _has_npu_device():
            from .npu_tf_backend import NpuTfBackend

            return NpuTfBackend()
        from .cpu_tf_backend import CpuTfBackend

        return CpuTfBackend()

    if framework == "onnx":
        if not force_cpu and _npu_onnx_available():
            from .onnx_backend import NpuOnnxBackend

            return NpuOnnxBackend()
        from .onnx_backend import CpuOnnxBackend

        return CpuOnnxBackend()

    if force_cpu:
        from .cpu_torch_backend import CpuTorchBackend

        return CpuTorchBackend()

    for name, profile in _hw_profiles("torch").items():
        # Skip CPU profiles during auto-detect (CPU is the fallback below).
        if profile.get("torch_lib") == "cpu":
            continue
        if _probe(profile):
            _log.info("Active hardware: %s (torch_lib=%s)", name, profile["torch_lib"])
            return _build("torch", name, profile)
    from .cpu_torch_backend import CpuTorchBackend

    return CpuTorchBackend()
