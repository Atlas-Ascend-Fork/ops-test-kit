# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""get_backend 路由测试 + _probe 硬件探测单元。

torch 路径：iterate _hw_profiles("torch")（frameworks 配置段），跳过 cpu
profile，_probe 命中即 _build；全部 miss → CPU 回落。tf 路径：npu_device
包安装探测。force_cpu 直取 CPU 系。
"""

from __future__ import annotations

import importlib

import ttk.core_modules.common.backend as be
from ttk.core_modules.common.backend import _probe, get_backend
from ttk.core_modules.common.backend.cpu_tf_backend import CpuTfBackend
from ttk.core_modules.common.backend.cpu_torch_backend import CpuTorchBackend


def test_force_cpu_returns_cpu_backend():
    assert isinstance(get_backend(force_cpu=True), CpuTorchBackend)
    assert isinstance(get_backend(force_cpu=True, framework="tf"), CpuTfBackend)


def test_torch_npu_profile_probe_success_returns_npu(monkeypatch):
    from ttk.core_modules.common.backend.npu_torch_backend import NpuTorchBackend

    monkeypatch.setattr(
        be,
        "_hw_profiles",
        lambda fw: {"npu": {"torch_lib": "npu", "profiler": "builtin"}} if fw == "torch" else {},
    )
    monkeypatch.setattr(be, "_probe", lambda p: True)
    backend = get_backend()
    assert isinstance(backend, NpuTorchBackend)
    # _build 注入 profile/segment（类属性等价物）：profiler 工厂与 alias 消费
    assert backend.profile == {"torch_lib": "npu", "profiler": "builtin"}
    assert backend.device_type() == "npu"
    assert backend.torch_lib == "npu"


def test_torch_xpu_profile_probe_success_returns_xpu(monkeypatch):
    from ttk.core_modules.common.backend.xpu_torch_backend import XpuTorchBackend

    profile = {"torch_lib": "cuda", "profiler": {"activities": ["CPU", "CUDA"]}}
    monkeypatch.setattr(be, "_hw_profiles", lambda fw: {"xpu": profile})
    monkeypatch.setattr(be, "_probe", lambda p: True)
    backend = get_backend()
    assert isinstance(backend, XpuTorchBackend)
    assert backend.torch_lib == "cuda"
    assert backend.profile == profile
    assert backend.device_type() == "xpu"  # alias = segment name


def test_torch_probe_failure_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(
        be,
        "_hw_profiles",
        lambda fw: {"npu": {"torch_lib": "npu", "profiler": "builtin"}},
    )
    monkeypatch.setattr(be, "_probe", lambda p: False)
    assert isinstance(get_backend(), CpuTorchBackend)


def test_torch_no_profiles_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(be, "_hw_profiles", lambda fw: {})
    assert isinstance(get_backend(), CpuTorchBackend)


def test_probe_cuda_skips_import(monkeypatch):
    """cuda native skips import: importlib.import_module must not be called."""
    called = []

    def spy(name):
        called.append(name)
        raise AssertionError("should not import for cuda")

    monkeypatch.setattr(importlib, "import_module", spy)
    _probe({"torch_lib": "cuda"})  # no raise + import not called
    assert called == []


def test_probe_missing_torch_lib_returns_false(caplog):
    """A profile missing torch_lib is rejected up front (not swallowed as
    "device not available"). Both empty {} and {profiler: ...} -> False +
    warning naming torch_lib."""
    for bad in ({}, {"profiler": "builtin"}):
        caplog.clear()
        assert _probe(bad) is False
        assert any("torch_lib" in rec.message for rec in caplog.records)


def test_probe_catches_runtime_error(monkeypatch):
    """PrivateUse1 single-slot RuntimeError caught by except."""

    def boom(name):
        raise RuntimeError("rename_privateuse1_backend already set")

    monkeypatch.setattr(importlib, "import_module", boom)
    assert _probe({"torch_lib": "mlu"}) is False


def test_tf_npu_device_probe(monkeypatch):
    from ttk.core_modules.common.backend.npu_tf_backend import NpuTfBackend

    monkeypatch.setattr(be, "_has_npu_device", lambda: True)
    assert isinstance(get_backend(framework="tf"), NpuTfBackend)
    monkeypatch.setattr(be, "_has_npu_device", lambda: False)
    assert isinstance(get_backend(framework="tf"), CpuTfBackend)
