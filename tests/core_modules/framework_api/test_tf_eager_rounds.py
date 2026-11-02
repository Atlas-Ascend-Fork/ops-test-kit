# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""TF eager 逐轮策略回归：run_tf_eager_rounds 的采集开关分支与
TfAdapter.eager_rounds 的关采集降级（修复前关采集委托基类
super().eager_rounds 抛 NotImplementedError）。纯 stub，不触达 tensorflow。"""

import contextlib
from types import SimpleNamespace

import numpy as np

from ttk.core_modules.framework_api.tf.eager_runner import run_tf_eager_rounds
from ttk.core_modules.framework_api.utils import RunCtx


class _Backend:
    @staticmethod
    def device_scope(_dev_id):
        return contextlib.nullcontext()

    @staticmethod
    def synchronize(_dev_id):
        return None

    @staticmethod
    def result_to_numpy(result):
        return [result]


class _Adapter:
    def __init__(self, output):
        self._output = output
        self.calls = 0

    def wrap_eager(self, resolved, api_name, backend):  # noqa: ARG002
        return resolved

    def eager_call(self, resolved, args, kwargs, is_tensor_method):  # noqa: ARG002
        self.calls += 1
        return self._output


class _Profiler:
    def __init__(self):
        self.steps = 0
        self.entered = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *exc):
        return False

    def step(self):
        self.steps += 1


def _ctx(task_profiling, run_count=2, warmup_count=0):
    switches = SimpleNamespace(
        task_profiling=task_profiling,
        force_clear_ub=None,
        force_clear_l1=None,
        force_clear_l0=None,
        mode=SimpleNamespace(is_model=lambda: True),
    )
    return RunCtx(
        backend=_Backend(),
        dev_id=0,
        switches=switches,
        resolved=object(),
        is_tensor_method=False,
        deterministic_level=0,
        run_count=run_count,
        warmup_count=warmup_count,
    )


def _rounds(ctx):
    adapter = _Adapter(np.array([1.0], dtype=np.float32))
    profiler = _Profiler()
    result, result_nps, md5_list = run_tf_eager_rounds(
        adapter,
        ctx,
        api_name="tf.raw_ops.InplaceAdd",
        args=[],
        kwargs={},
        inplace_input_indexes=(),
        mutable_loader=lambda _i: None,
        profiler=profiler,
        backward_mode=False,
    )
    return adapter, profiler, result, result_nps, md5_list


def test_profiling_off_single_phase():
    """关采集：仅正确性轮（run_count 次调用），无计时段双倍执行。"""
    adapter, profiler, result, result_nps, md5_list = _rounds(_ctx(False, run_count=3))
    assert adapter.calls == 3
    assert profiler.entered == 0
    assert profiler.steps == 0
    assert result.tolist() == [1.0]
    assert len(result_nps) == 1
    assert result_nps[0].tolist() == [1.0]
    assert md5_list == []


def test_profiling_on_two_phases():
    """开采集：正确性轮 + warmup 轮 + 计时段（窗口内不复位）。"""
    adapter, profiler, result, result_nps, md5_list = _rounds(_ctx(True, run_count=2, warmup_count=1))
    assert adapter.calls == 2 + 1 + 2
    assert profiler.steps == 1
    assert profiler.entered == 1
    assert len(result_nps) == 1


def test_tf_adapter_eager_rounds_delegates_when_profiling_off(monkeypatch):
    """TfAdapter.eager_rounds 关采集走 run_tf_eager_rounds（修复前 super()
    抛 NotImplementedError）。"""
    import ttk.core_modules.framework_api.tf.eager_runner as eager_runner_mod
    from ttk.core_modules.framework_api.adapter import get_adapter

    marker = object()
    captured = {}

    def fake_rounds(_adapter, _ctx, **kwargs):
        captured.update(kwargs)
        return marker

    monkeypatch.setattr(eager_runner_mod, "run_tf_eager_rounds", fake_rounds)
    out = get_adapter("tf").eager_rounds(
        _ctx(False),
        api_name="tf.raw_ops.InplaceAdd",
        args=[],
        kwargs={},
        inplace_input_indexes=(),
        mutable_loader=lambda _i: None,
        profiler=_Profiler(),
        backward_mode=False,
    )
    assert out is marker
    assert captured["api_name"] == "tf.raw_ops.InplaceAdd"
