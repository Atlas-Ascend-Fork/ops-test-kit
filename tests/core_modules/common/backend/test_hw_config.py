# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""hardware config in default.yaml + get_hardware_config + profile validation.

- frameworks.torch.npu segment ships in default.yaml (torch_lib=npu, profiler=builtin).
- get_hardware_config() returns {} when config empty (only-cpu legal default;
  distinct from remote's None).
- _build(fw, name, profile) fail-fast validates torch_lib + profiler before
  instantiating/injecting the backend.
"""

from __future__ import annotations

import pytest

from ttk.config.loader import get_hardware_config, load_config
from ttk.core_modules.common.backend import _build


def test_hardware_config_returns_frameworks():
    load_config()
    hw = get_hardware_config()
    assert hw["torch"]["npu"]["torch_lib"] == "npu"


def test_get_hardware_config_empty_returns_dict(monkeypatch):
    monkeypatch.setattr("ttk.config.loader.get_config", dict)
    assert get_hardware_config() == {}


def test_build_missing_torch_lib_raises():
    with pytest.raises(ValueError, match="torch_lib"):
        _build("torch", "npu", {"profiler": "builtin"})


def test_build_invalid_profiler_raises():
    with pytest.raises(ValueError, match="profiler"):
        _build("torch", "xpu", {"torch_lib": "cuda", "profiler": 123})  # 非 builtin/dict


def test_build_builtin_non_npu_raises():
    # builtin 是 NPU 专用；非 npu 会路由 TorchProfiler 按 activities 消费，运行时 TypeError
    with pytest.raises(ValueError, match="builtin"):
        _build("torch", "xpu", {"torch_lib": "cuda", "profiler": "builtin"})


def test_build_injects_profile_and_segment():
    from ttk.core_modules.common.backend.xpu_torch_backend import XpuTorchBackend

    profile = {"torch_lib": "cuda", "profiler": {"activities": ["CPU", "CUDA"]}}
    b = _build("torch", "xpu", profile)
    assert isinstance(b, XpuTorchBackend)
    assert b.torch_lib == "cuda"
    assert b.profile == profile
    assert b._segment_name == "xpu"


def test_xpu_set_device(monkeypatch):
    """set_device 按 torch_lib 派生：隐式落卡路径（无 device 参数的 op）走被分配的卡。"""
    import torch

    from ttk.core_modules.common.backend.xpu_torch_backend import XpuTorchBackend

    class FakeMlu:
        @staticmethod
        def set_device(dev_id):
            FakeMlu.called = dev_id

    monkeypatch.setattr(torch, "mlu", FakeMlu, raising=False)
    backend = XpuTorchBackend()
    backend.torch_lib = "mlu"
    backend.set_device(3)
    assert FakeMlu.called == 3
