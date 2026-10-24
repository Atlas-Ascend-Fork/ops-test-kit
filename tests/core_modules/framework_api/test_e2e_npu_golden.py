#!/usr/bin/env python3 -*-
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""e2e `_generate_npu_golden_with_perf` 测试（MsProfiler 全程 mock，不触设备）。

- NPU 执行失败时回落 CPU 生成，source="cpu"
- golden_api 在 NPU 模式下于设备上执行（exec_backend 注入），source="npu"
"""

from types import SimpleNamespace

import numpy as np

import ttk.core_modules.common.golden.npu_golden as _npu_golden_module
from ttk.core_modules.framework_api.golden import _generate_npu_golden_with_perf
from ttk.core_modules.framework_api.result import FrameworkApiReturnStructure


class _FakeMsProfiler:
    def __init__(self, dev_id, result_path=None, prof_type=None, start_step=1):
        self.dev_id = dev_id
        self.start_step = start_step
        self.steps = 0

    def __enter__(self):
        return self

    def step(self):
        self.steps += 1

    def __exit__(self, *exc):
        return False

    def result(self, repeat_count=1):
        from ttk.core_modules.common.profiler.base import DisabledProfiler

        return DisabledProfiler().result(repeat_count)


def _switches(tmp_path):
    return SimpleNamespace(
        golden_mode="NPU", plugin_path=None, root_path=str(tmp_path), task_profiling=True, warmup=False
    )


def _backend():
    return SimpleNamespace(device_type=lambda: "npu")


def _case(golden_api=None):
    return SimpleNamespace(
        golden_mode_override=None,
        golden_api=golden_api,
        golden_perf=None,
        golden_source=None,
        testcase_name="case_t",
    )


def _patch_generate_golden(monkeypatch, calls):
    def _fake_generate_golden(
        testcase, raw_inputs, plugin_path=None, switches=None, backend="cpu", exec_backend=None, dev_id=0
    ):
        calls.append(exec_backend is not None)
        if exec_backend is not None:
            raise RuntimeError("npu device error")
        return [np.array([2.0])]

    monkeypatch.setattr("ttk.core_modules.framework_api.golden.generate_golden", _fake_generate_golden)


def test_npu_execution_failure_falls_back_to_cpu(monkeypatch, tmp_path):
    """NPU 执行失败 → 回落 CPU 生成真值，source="cpu"。"""
    monkeypatch.setattr(_npu_golden_module, "MsProfiler", _FakeMsProfiler)
    monkeypatch.setattr("ttk.core_modules.common.golden.has_npu_device", lambda: True)
    calls = []
    _patch_generate_golden(monkeypatch, calls)

    return_struct = FrameworkApiReturnStructure()
    golden_nps = _generate_npu_golden_with_perf(_case(), [], _switches(tmp_path), _backend(), 0, return_struct)

    assert calls == [True, False], f"expected NPU attempt then CPU fallback, got {calls}"
    np.testing.assert_array_equal(golden_nps[0], np.array([2.0]))
    assert return_struct.golden_source == "cpu"
    assert return_struct.golden_device_perf_us == "----"


def test_golden_api_runs_on_npu_through_window(monkeypatch, tmp_path):
    """golden_api 在 NPU 模式下于设备上执行（进采集窗，无 CPU 回落），
    source="npu"。"""
    monkeypatch.setattr("ttk.core_modules.common.golden.has_npu_device", lambda: True)
    monkeypatch.setattr(_npu_golden_module, "MsProfiler", _FakeMsProfiler)
    calls = []

    def _fake_generate_golden(
        testcase, raw_inputs, plugin_path=None, switches=None, backend="cpu", exec_backend=None, dev_id=0
    ):
        calls.append(exec_backend is not None)
        return [np.array([2.0])]

    monkeypatch.setattr("ttk.core_modules.framework_api.golden.generate_golden", _fake_generate_golden)

    return_struct = FrameworkApiReturnStructure()
    golden_nps = _generate_npu_golden_with_perf(
        _case(golden_api="torch.add"), [], _switches(tmp_path), _backend(), 0, return_struct
    )

    assert calls == [True], f"golden_api must run via the NPU form, got {calls}"
    np.testing.assert_array_equal(golden_nps[0], np.array([2.0]))
    assert return_struct.golden_source == "npu"
    assert return_struct.golden_device_perf_us == "----"
