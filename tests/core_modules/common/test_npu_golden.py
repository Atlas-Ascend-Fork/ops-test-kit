# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""npu_golden_runner 回落语义测试：task_profiling 开关对称回落、CPU 占位
根因注解、CPU 闭包失败上抛。MsProfiler 全程 mock，不触设备。"""

from types import SimpleNamespace

import numpy
import pytest

from ttk.core_modules.common.golden import npu_golden as _module
from ttk.core_modules.common.golden import npu_golden_runner


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


@pytest.fixture
def switches(tmp_path):
    return SimpleNamespace(root_path=str(tmp_path), task_profiling=True, warmup=False)


def _switches_no_profiling(tmp_path):
    return SimpleNamespace(root_path=str(tmp_path), task_profiling=False, warmup=False)


def test_profiling_disabled_runs_npu_once_without_window(monkeypatch, tmp_path):
    """task_profiling=False：裸执行一次，不开采集窗，source="npu"。"""
    calls = []

    def _no_profiler(*_args, **_kwargs):
        raise AssertionError("MsProfiler must not be constructed when profiling is disabled")

    monkeypatch.setattr(_module, "MsProfiler", _no_profiler)
    outputs, perf, source = npu_golden_runner(
        lambda: calls.append("npu") or [numpy.ones(1)],
        lambda: (_ for _ in ()).throw(AssertionError("cpu called")),
        0,
        _switches_no_profiling(tmp_path),
        "case",
    )
    assert calls == ["npu"]
    assert source == "npu"
    assert perf.elapsed_us <= 0
    assert len(outputs) == 1


def test_profiling_disabled_npu_failure_falls_back_to_cpu(monkeypatch, tmp_path):
    """task_profiling=False 时 NPU 执行失败同样回落 CPU（与采集开启对称）。"""
    monkeypatch.setattr(_module, "MsProfiler", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("no profiler")))

    def _invoke_npu():
        raise RuntimeError("sim device unavailable")

    outputs, _perf, source = npu_golden_runner(
        _invoke_npu, lambda: [numpy.array([1.0])], 0, _switches_no_profiling(tmp_path), "case"
    )
    assert source == "cpu"
    assert len(outputs) == 1


def test_npu_failure_falls_back_to_cpu_with_real_golden(monkeypatch, switches):
    """采集窗内 NPU 失败 → CPU 真值可用：输出不注解、source="cpu"。"""
    monkeypatch.setattr(_module, "MsProfiler", _FakeMsProfiler)
    golden = [numpy.array([2.0])]
    outputs, _perf, source = npu_golden_runner(
        lambda: (_ for _ in ()).throw(RuntimeError("device error")), lambda: golden, 0, switches, "case"
    )
    assert source == "cpu"
    assert outputs == golden


def test_npu_failure_with_cpu_placeholders_annotates_cause(monkeypatch, switches):
    """CPU 回落全占位（NPU-only 算子）→ 占位串追加 NPU 根因，失败行可诊断。"""
    monkeypatch.setattr(_module, "MsProfiler", _FakeMsProfiler)
    outputs, _perf, source = npu_golden_runner(
        lambda: (_ for _ in ()).throw(RuntimeError("aclrtSetDevice failed")),
        lambda: ["UNSUPPORTED"],
        0,
        switches,
        "case",
    )
    assert source == "cpu"
    assert outputs[0].startswith("UNSUPPORTED (NPU golden failed:")
    assert "aclrtSetDevice failed" in outputs[0]


def test_cpu_fallback_failure_propagates(monkeypatch, switches):
    """CPU 闭包自身失败：原样上抛（NPU 根因经 error 日志承载，不吞）。"""
    monkeypatch.setattr(_module, "MsProfiler", _FakeMsProfiler)

    def _invoke_cpu():
        raise ValueError("cpu golden bug")

    with pytest.raises(ValueError, match="cpu golden bug"):
        npu_golden_runner(lambda: (_ for _ in ()).throw(RuntimeError("device error")), _invoke_cpu, 0, switches, "case")


def test_suppressed_or_mixed_outputs_not_annotated(monkeypatch, switches):
    """SUPPRESSED（PASS-SKIP 哨兵）或含真实数组的混合输出保持原样。"""
    monkeypatch.setattr(_module, "MsProfiler", _FakeMsProfiler)
    outputs, _, source = npu_golden_runner(
        lambda: (_ for _ in ()).throw(RuntimeError("device error")),
        lambda: ["SUPPRESSED"],
        0,
        switches,
        "case",
    )
    assert outputs == ["SUPPRESSED"]

    mixed = [numpy.array([1.0]), "GOLDEN_FAILURE"]
    outputs, _, _ = npu_golden_runner(
        lambda: (_ for _ in ()).throw(RuntimeError("device error")), lambda: mixed, 0, switches, "case"
    )
    assert outputs == mixed
