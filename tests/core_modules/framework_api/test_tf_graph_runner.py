# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""tf graph_runner 两阶段 mutable 执行时序看护。

无输出有状态算子的回读取变量当前值：Phase 2（profiling 窗口内不复位）会在
Phase 1 结果之上继续累积状态，因此比对结果必须在 Phase 1 末尾定格。修复前
result_nps 在 Phase 2 之后才回读，读到 init + (run_count+1)·delta，与 golden
（init + delta）比对必误报。conftest 的 _framework_preload shim 已保证本进程
内 tensorflow 可用（RTLD_DEEPBIND）。"""

from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from ttk.core_modules.framework_api.tf.graph_runner import execute_tf_graph  # noqa: E402


class _FakeBackend:
    tf_device_type = "CPU"

    def is_npu(self):
        return False

    def to_device(self, t, dev_id, preserve_stride=True):
        return t

    def needs_numpy_fallback(self, testcase):
        return False

    def result_to_numpy(self, result, copy=False):
        if result is None:
            return [None]
        results = result if isinstance(result, (tuple, list)) else [result]
        return [r.numpy() if hasattr(r, "numpy") else np.asarray(r) for r in results]

    def synchronize(self, dev_id):
        pass


class _FakeProfiler:
    def __init__(self):
        self.windows = 0

    def __enter__(self):
        self.windows += 1
        return self

    def __exit__(self, *exc):
        return False

    def result(self, run_count):
        return f"perf:{run_count}"


class _FakePlan:
    def build_args(self, tensors):
        return list(tensors), {}, None


def _switches(tmp_path, run_time=3):
    return SimpleNamespace(
        task_profiling=True,
        run_time=run_time,
        warmup=False,
        root_path=str(tmp_path),
        plugin_path=None,
    )


def test_no_output_mutable_readback_pinned_to_phase1(tmp_path, monkeypatch):
    """核心看护：Phase 2 照跑（计时窗口不可省），但 result_nps 必须定格在
    Phase 1 末尾（恰一次复位后执行的结果），不被窗口内累积状态污染。"""
    import ttk.core_modules.common.profiler as prof_mod
    import ttk.core_modules.framework_api.tf.param_extractor as pex

    calls = {"n": 0}

    def fake_stateful_op(var):
        calls["n"] += 1
        var.assign_add(1.0)

    def fake_collect(api_name, args=None, kwargs=None):
        return [v for v in (args or ()) if isinstance(v, tf.Variable)]

    profiler = _FakeProfiler()
    monkeypatch.setattr(pex, "collect_mutable_variables", fake_collect)
    monkeypatch.setattr(prof_mod, "get_profiler", lambda *a, **k: profiler)

    var = tf.Variable(0.0)
    tc = SimpleNamespace(
        api_name="tf.raw_ops.FakeStateful",
        testcase_name="case_stateful",
        tensors=[var],
        flat_tensors=[var],
        tensor_list_dist=(),
        flat_tensor_view_shapes=(),
        flat_tensor_dtypes=(),
        const_input_indexes=set(),
        dyn_input_slot_indexes=(),
    )
    raw_inputs = [np.zeros((), dtype=np.float32)]

    result_nps, perf, det_status, _err = execute_tf_graph(
        tc,
        _FakeBackend(),
        0,
        _switches(tmp_path),
        _FakePlan(),
        fake_stateful_op,
        False,
        False,
        raw_inputs,
        dynamic=False,
        deterministic_level=0,
    )

    # 恰一次 trace（assign_add 图执行 6 次：3 正确性 + 3 计时），变量累积到 4.0
    assert calls["n"] == 1
    assert var.numpy() == 4.0
    # 但比对结果必须是恰一次执行后的 1.0（修复前此处读到 4.0）
    assert result_nps == [1.0]
    assert perf == "perf:3"
    assert det_status is None
    assert profiler.windows == 1


def test_output_bearing_result_from_last_run(tmp_path, monkeypatch):
    """有返回值 API（无 mutable 位）走单阶段路径，结果取末轮返回值。"""
    import ttk.core_modules.common.profiler as prof_mod

    counter = tf.Variable(0)

    def fake_op(x):
        counter.assign_add(1)
        return x * 2

    profiler = _FakeProfiler()
    monkeypatch.setattr(prof_mod, "get_profiler", lambda *a, **k: profiler)

    tc = SimpleNamespace(
        api_name="tf.fake.Scale",
        testcase_name="case_scale",
        tensors=None,
        tensor_list_dist=(),
        flat_tensor_view_shapes=(),
        flat_tensor_dtypes=(),
        const_input_indexes=set(),
        dyn_input_slot_indexes=(),
    )
    raw_inputs = [np.ones(3, dtype=np.float32)]

    result_nps, perf, det_status, _err = execute_tf_graph(
        tc,
        _FakeBackend(),
        0,
        _switches(tmp_path),
        _FakePlan(),
        fake_op,
        False,
        False,
        raw_inputs,
        dynamic=False,
        deterministic_level=0,
    )

    # 图执行 3 次（闭包计数变量被 capture，随每次调用累积）
    assert counter.numpy() == 3
    assert result_nps[0].tolist() == [2.0, 2.0, 2.0]
    assert perf == "perf:3"
    assert det_status is None
    assert profiler.windows == 1


def test_unmappable_dtype_fails_leg(tmp_path, monkeypatch):
    """dtype 无法映射 tf.dtype 时必须显式失败（[] 返回），
    不得静默降级为无签名自动 trace（静态模式语义变动态、对外仍报 static）。"""
    import ttk.core_modules.common.profiler as prof_mod

    def fake_op(x):
        return x

    profiler = _FakeProfiler()
    monkeypatch.setattr(prof_mod, "get_profiler", lambda *a, **k: profiler)

    tc = SimpleNamespace(
        api_name="tf.fake.Scale",
        testcase_name="case_bogus_dtype",
        tensors=None,
        tensor_list_dist=(),
        flat_tensor_view_shapes=((3,),),
        flat_tensor_dtypes=("not_a_dtype",),
        const_input_indexes=set(),
        dyn_input_slot_indexes=(),
    )
    raw_inputs = [np.ones(3, dtype=np.float32)]

    result_nps, perf, det_status, _err = execute_tf_graph(
        tc,
        _FakeBackend(),
        0,
        _switches(tmp_path),
        _FakePlan(),
        fake_op,
        False,
        False,
        raw_inputs,
        dynamic=False,
        deterministic_level=0,
    )

    assert (result_nps, perf, det_status) == ([], None, None)
    assert _err is not None
    assert "TF graph static execution failed" in _err
    assert "not_a_dtype" in _err
    assert profiler.windows == 0


class TestBuildInputSignature:
    """complex32 → fp16 + 尾维 [2] lanes 签名（与 eager 喂入形态一致）。"""

    @staticmethod
    def _testcase(shapes, dtypes):
        return SimpleNamespace(
            flat_tensor_view_shapes=shapes,
            flat_tensor_dtypes=dtypes,
            const_input_indexes=set(),
            tensor_list_dist=(),
            api_name="tf.raw_ops.InplaceAdd",
        )

    @pytest.mark.parametrize(
        ("dtype", "dynamic", "expected"),
        [
            pytest.param("complex32", False, (2, 3, 2), id="static"),
            pytest.param("complex32", True, (None, None, 2), id="dynamic"),
            pytest.param("c32", False, (2, 3, 2), id="alias"),
        ],
    )
    def test_complex32_maps_to_fp16_lanes(self, dtype, dynamic, expected):
        from ttk.core_modules.framework_api.tf.graph_runner import _build_input_signature

        sig, sig_idx = _build_input_signature(self._testcase(((2, 3),), (dtype,)), dynamic=dynamic)
        assert sig_idx == [0]
        assert sig[0].shape == expected
        assert sig[0].dtype == tf.float16

    def test_native_dtype_unchanged(self):
        from ttk.core_modules.framework_api.tf.graph_runner import _build_input_signature

        sig, _ = _build_input_signature(self._testcase(((2, 3), (1,)), ("float32", "int32")), dynamic=False)
        assert sig[0].shape == (2, 3)
        assert sig[0].dtype == tf.float32
        assert sig[1].shape == (1,)
        assert sig[1].dtype == tf.int32
