#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""input_generate 白盒 UT：默认输入生成（_default_generate_inputs）、CPU 放置
作用域（_cpu_placement_scope）、customize_inputs 返回值回写（_apply_input_plugin_result/
_plugin_value_to_numpy）、requires_grad 回放（set_runtime_tensors）。"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ttk.core_modules.framework_api.input_generate import (
    _apply_input_plugin_result,
    _cpu_placement_scope,
    _default_generate_inputs,
    _plugin_value_to_numpy,
    set_runtime_tensors,
)


def _make_switches(**overrides):
    """构造测试用 SWITCHES 对象，默认 input_distribution='uniform'。"""
    from ttk.switches import SWITCHES

    sw = SWITCHES()
    sw.input_distribution = "uniform"
    for k, v in overrides.items():
        setattr(sw, k, v)
    return sw


def _make_testcase(
    shapes,
    dtypes,
    attrs=None,
    output_tensor_indexes=(),
    pure_output_indexes=None,
    storage_shapes=(),
    view_strides=(),
    view_offsets=(),
    input_data_ranges=None,
):
    """构造一个 MagicMock 测试用例，模拟 flat_tensor_* 系列接口。"""
    case = MagicMock()
    case.flat_tensor_view_shapes = shapes
    case.flat_tensor_dtypes = dtypes
    case.flat_input_data_ranges = input_data_ranges or ()
    case.pure_output_indexes = pure_output_indexes or []
    case.flat_tensor_storage_shapes = storage_shapes
    case.flat_tensor_view_strides = view_strides
    case.flat_tensor_view_offsets = view_offsets

    def flat_storage_side_effect(idx):
        if storage_shapes and idx < len(storage_shapes):
            val = storage_shapes[idx]
            if val is not None:
                return val
        return shapes[idx] if idx < len(shapes) else None

    def flat_stride_side_effect(idx):
        if view_strides and idx < len(view_strides):
            s = view_strides[idx]
            if s is not None and s != ():
                return s
        from ttk.utilities.shape import shape_stride

        v = shapes[idx] if idx < len(shapes) else None
        return shape_stride(v) if v is not None else None

    def flat_offset_side_effect(idx):
        if view_offsets and idx < len(view_offsets):
            val = view_offsets[idx]
            if val is not None:
                return val
        return 0

    case.flat_storage_shape.side_effect = flat_storage_side_effect
    case.flat_view_stride.side_effect = flat_stride_side_effect
    case.flat_view_offset.side_effect = flat_offset_side_effect
    return case


# --- 默认输入生成 ---


class TestDefaultGenerateInputs:
    """_default_generate_inputs: 默认输入张量生成逻辑。"""

    def test_basic_input_generation(self):
        """基本场景：两个 float32 张量正确生成。"""
        case = _make_testcase(shapes=((4,), (4,)), dtypes=("float32", "float32"))
        switches = _make_switches()
        inputs = _default_generate_inputs(case, switches)
        assert len(inputs) == 2
        assert inputs[0].shape == (4,)
        assert inputs[0].dtype == np.float32
        assert inputs[1].shape == (4,)
        assert inputs[1].dtype == np.float32

    def test_complex32_input_keeps_lane_storage(self):
        """complex32：视图即存储（fp16 + 尾维 2），不按声明形状坍缩成半数据。"""
        case = _make_testcase(shapes=((2, 3),), dtypes=("complex32",))
        inputs = _default_generate_inputs(case, _make_switches())
        assert inputs[0].shape == (2, 3, 2)
        assert inputs[0].dtype == np.float16


# --- CPU 放置作用域 ---


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


def test_set_runtime_tensors_threads_scope(monkeypatch):
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


# --- customize_inputs 返回值回写 ---


def _nb():
    return SimpleNamespace(testcase_name="case_x")


def test_none_keeps_inplace_semantics():
    """整体 None 或槽位 None：不回写，保持生成值。"""
    raw = [np.ones(2, dtype=np.float32), np.zeros(2, dtype=np.int32)]
    _apply_input_plugin_result(None, raw, _nb(), None)
    _apply_input_plugin_result((np.array([3.0, 4.0], dtype=np.float32), None), raw, _nb(), None)
    assert raw[0].tolist() == [3.0, 4.0]
    assert raw[1].tolist() == [0, 0]


def test_return_writes_back():
    """返回序列按槽位写回：视图写穿 storage；TensorList 嵌套展平；单张量可裸返回。"""
    storage = np.zeros(8, dtype=np.float32)
    _apply_input_plugin_result((np.arange(4, dtype=np.float32),), [storage[::2]], _nb(), None)
    assert storage.tolist() == [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0]

    raw = [np.zeros(2), np.zeros(2), np.zeros(2)]
    _apply_input_plugin_result(([np.ones(2), np.full(2, 2.0)], np.full(2, 3.0)), raw, _nb(), None)
    assert [r.tolist() for r in raw] == [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]

    single = [np.zeros(3, dtype=np.float32)]
    _apply_input_plugin_result(np.array([1.0, 2.0, 3.0], dtype=np.float32), single, _nb(), None)
    assert single[0].tolist() == [1.0, 2.0, 3.0]


def test_mismatch_raises():
    """长度/shape 不符、dynamic 槽返回非 None：显式报错。"""
    raw2 = [np.zeros(2), np.zeros(2)]
    with pytest.raises(RuntimeError, match="returned 1 tensor"):
        _apply_input_plugin_result((np.zeros(2),), raw2, _nb(), None)

    with pytest.raises(RuntimeError, match=r"shape \(3, 2\) for flat slot 0, expected \(2, 3\)"):
        _apply_input_plugin_result((np.zeros((3, 2)),), [np.zeros((2, 3))], _nb(), None)

    with pytest.raises(RuntimeError, match="no generated tensor"):
        _apply_input_plugin_result((np.zeros(2), None), [None, np.zeros(2)], _nb(), None)


def test_plugin_value_to_numpy():
    """框架张量/普通对象统一转 numpy。"""
    tf = pytest.importorskip("tensorflow")

    from ttk.core_modules.common.backend.cpu_tf_backend import CpuTfBackend

    backend = CpuTfBackend()
    t = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    arr = _plugin_value_to_numpy(t, backend)
    assert isinstance(arr, np.ndarray)
    assert arr.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert _plugin_value_to_numpy(tf.Variable([5.0, 6.0]), backend).tolist() == [5.0, 6.0]
    assert _plugin_value_to_numpy(np.array([1.5]), backend).tolist() == [1.5]
    assert _plugin_value_to_numpy([1, 2, 3], backend).tolist() == [1, 2, 3]


# --- requires_grad 回放（set_runtime_tensors） ---


class _ReplayTensor:
    def __init__(self, requires_grad=False):
        self.requires_grad = requires_grad

    def requires_grad_(self, value):
        self.requires_grad = value


class _ReplayBackend:
    def inputs_from_numpy(self, testcase, raw_inputs):
        return [_ReplayTensor() for _ in raw_inputs]


def test_plugin_requires_grad_replayed_onto_rebuilt_tensors():
    """插件置的 requires_grad 回放到重建张量；插件对象不进 testcase.tensors。"""
    testcase = SimpleNamespace(tensor_list_dist=None)
    plugin_tensors = [_ReplayTensor(requires_grad=True), None, np.zeros(2)]
    set_runtime_tensors(testcase, [0, 1, 2], _ReplayBackend(), plugin_tensors=plugin_tensors)

    rebuilt = testcase.tensors
    assert rebuilt[0].requires_grad is True
    assert rebuilt[1].requires_grad is False
    assert rebuilt[2].requires_grad is False
    assert all(t is not plugin_tensors[0] for t in rebuilt)


def test_plugin_requires_grad_replay_truncates_to_shorter_list():
    """回放按较短的 flat_tensors 截断，长度失配不越界。"""
    testcase = SimpleNamespace(tensor_list_dist=None)
    plugin_tensors = [_ReplayTensor(requires_grad=True), _ReplayTensor(requires_grad=True)]
    set_runtime_tensors(testcase, [0], _ReplayBackend(), plugin_tensors=plugin_tensors)
    assert testcase.tensors[0].requires_grad is True

    set_runtime_tensors(testcase, [0, 1, 2], _ReplayBackend(), plugin_tensors=[])
    assert all(t.requires_grad is False for t in testcase.tensors)
