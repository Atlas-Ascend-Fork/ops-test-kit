# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""输入物化的设备作用域分派测试：cpu_placement 只对 TF 后端启用 CPU 放置。

流程：输入生成/golden 在 CPU 作用域物化（TF），backend open 后设备张量
随默认设备（NPU）重建。torch 侧张量本就 CPU-born，不进作用域（也不得触达
tensorflow import）。
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from ttk.core_modules.framework_api.input_generate import _cpu_placement_scope


class _FakeTfBackend:
    tf_device_type = "npu"


class _FakeTorchBackend:
    pass


def test_cpu_scope_applies_to_tf_backend_when_enabled():
    entered = []

    class _scope:
        def __enter__(self):
            entered.append("cpu")

        def __exit__(self, *exc):
            return False

    with patch("ttk.core_modules.common.backend.cpu_tf_backend.cpu_device_context", return_value=_scope()):
        scope = _cpu_placement_scope(_FakeTfBackend(), enabled=True)
        with scope:
            pass
    assert entered == ["cpu"]


def test_cpu_scope_skipped_for_tf_when_disabled():
    calls = []
    with patch(
        "ttk.core_modules.common.backend.cpu_tf_backend.cpu_device_context",
        side_effect=lambda: calls.append(1),
    ):
        scope = _cpu_placement_scope(_FakeTfBackend(), enabled=False)
        assert isinstance(scope, nullcontext)
        with scope:
            pass
    assert calls == []


def test_cpu_scope_skipped_for_torch_backend():
    """torch 后端不进作用域：即使 enabled 也不触达 tensorflow 路径。"""
    calls = []
    with patch(
        "ttk.core_modules.common.backend.cpu_tf_backend.cpu_device_context",
        side_effect=lambda: calls.append(1),
    ):
        scope = _cpu_placement_scope(_FakeTorchBackend(), enabled=True)
        assert isinstance(scope, nullcontext)
    assert calls == []


def testset_runtime_tensors_threads_scope(monkeypatch):
    """set_runtime_tensors 把 cpu_placement 传给作用域分派并写回 testcase.tensors。"""
    from ttk.core_modules.framework_api import input_generate

    recorder = []

    class _scope:
        def __init__(self, enabled):
            recorder.append(enabled)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(input_generate, "_cpu_placement_scope", lambda backend, enabled: _scope(enabled))
    monkeypatch.setattr(
        _FakeTfBackend,
        "inputs_from_numpy",
        lambda self, testcase, raw_inputs: [f"t{i}" for i in raw_inputs],
        raising=False,
    )
    backend = _FakeTfBackend()
    testcase = SimpleNamespace(tensor_list_dist=None)

    input_generate.set_runtime_tensors(testcase, [0, 1], backend, cpu_placement=True)
    assert testcase.tensors == ["t0", "t1"]

    input_generate.set_runtime_tensors(testcase, [0], backend, cpu_placement=False)
    assert testcase.tensors == ["t0"]
    assert recorder == [True, False]
