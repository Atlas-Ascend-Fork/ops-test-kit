# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""get_backend 路由测试：framework × (CPU | NPU) 二叉决策。

torch：_npu_torch_available 探测（import torch_npu + torch.npu.is_available，
失败告警回落）；tf：npu_device 包安装探测。force_cpu 直取 CPU 系。
"""

from __future__ import annotations

import ttk.core_modules.common.backend as be
from ttk.core_modules.common.backend import get_backend
from ttk.core_modules.common.backend.cpu_tf_backend import CpuTfBackend
from ttk.core_modules.common.backend.cpu_torch_backend import CpuTorchBackend


def test_force_cpu_returns_cpu_backend():
    assert isinstance(get_backend(force_cpu=True), CpuTorchBackend)
    assert isinstance(get_backend(force_cpu=True, framework="tf"), CpuTfBackend)


def test_torch_npu_probe_failure_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(be, "_npu_torch_available", lambda: False)
    backend = get_backend()
    assert isinstance(backend, CpuTorchBackend)


def test_torch_npu_probe_success_returns_npu(monkeypatch):
    from ttk.core_modules.common.backend.npu_torch_backend import NpuTorchBackend

    monkeypatch.setattr(be, "_npu_torch_available", lambda: True)
    backend = get_backend()
    assert isinstance(backend, NpuTorchBackend)
    # _build 注入等价物改为类属性：profiler 工厂与 alias 消费
    assert backend.profile == {"torch_lib": "npu", "profiler": "builtin"}
    assert backend.device_type() == "npu"
    assert backend.torch_lib == "npu"


def test_tf_npu_device_probe(monkeypatch):
    from ttk.core_modules.common.backend.npu_tf_backend import NpuTfBackend

    monkeypatch.setattr(be, "_has_npu_device", lambda: True)
    assert isinstance(get_backend(framework="tf"), NpuTfBackend)
    monkeypatch.setattr(be, "_has_npu_device", lambda: False)
    assert isinstance(get_backend(framework="tf"), CpuTfBackend)
