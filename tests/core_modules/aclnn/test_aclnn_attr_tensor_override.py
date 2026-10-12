#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""aclnn InputGenerator._override_tensors_from_attributes 单元测试。

覆盖：
- 标量/数组值按张量形参名覆写对应槽位，已消费键从 attributes 移除
- TensorList 标量广播与逐子张量对位
- 非张量键保留；None 值显式报错；动态槽（无 storage）告警跳过
- 无 attributes / 未知 API 的空操作
"""

from types import SimpleNamespace

import numpy as np
import pytest

from ttk.core_modules.aclnn import input_generate as aclnn_input_generate
from ttk.core_modules.aclnn.input_generate import InputGenerator
from ttk.utilities import shape_stride


def _fake_keeper(tensor_names):
    """OpApiInfoKeeper 桩：仅 aclnnTest 有 info，tensors 为给定形参名列表。"""

    class _Info:
        tensors = list(tensor_names)

    class _Keeper:
        def info_of(self, api_name):
            return _Info() if api_name == "aclnnTest" else None

    return _Keeper


def _stub_case(attributes, storages, view_shapes, dist=(0, 0)):
    return SimpleNamespace(
        api_name="aclnnTest",
        testcase_name="t",
        attributes=attributes,
        tensor_list_dist=tuple(dist),
        np_storages=storages,
        flat_tensor_view_shapes=tuple(view_shapes),
        flat_view_stride=lambda idx: shape_stride(view_shapes[idx]) if view_shapes[idx] is not None else None,
        flat_view_offset=lambda idx: 0,
    )


def _rand(shape, dtype=np.float32):
    return np.random.uniform(-2, 2, shape).astype(dtype)


def test_scalar_value_overrides_slot_and_consumes_key(monkeypatch):
    """标量值覆写对应槽位（写穿 storage），键从 attributes 移除、非张量键保留。"""
    monkeypatch.setattr(aclnn_input_generate, "OpApiInfoKeeper", _fake_keeper(["x", "y"]))
    attrs = {"x": 1.0, "keep": 3}
    storages = [_rand((2, 3)), _rand((4,))]
    case = _stub_case(attrs, storages, ((2, 3), (4,)))

    InputGenerator(case)._override_tensors_from_attributes()

    np.testing.assert_array_equal(storages[0], np.full((2, 3), 1.0, np.float32))
    assert not np.allclose(storages[1], 2.0)  # 未声明的槽不受影响（仍为随机值）
    assert case.attributes == {"keep": 3}


def test_array_value_overrides_slot(monkeypatch):
    """数组值按 view shape 对位覆写。"""
    monkeypatch.setattr(aclnn_input_generate, "OpApiInfoKeeper", _fake_keeper(["x", "y"]))
    val = [[1.0, 2.0], [3.0, 4.0]]
    storages = [_rand((2, 2)), _rand((4,))]
    case = _stub_case({"x": val}, storages, ((2, 2), (4,)))

    InputGenerator(case)._override_tensors_from_attributes()

    np.testing.assert_array_equal(storages[0], np.array(val, np.float32))
    assert "x" not in case.attributes


def test_tensorlist_scalar_broadcast(monkeypatch):
    """TensorList 标量广播：同一值写到全部子张量。"""
    monkeypatch.setattr(aclnn_input_generate, "OpApiInfoKeeper", _fake_keeper(["x", "y"]))
    storages = [_rand((2,)), _rand((3,)), _rand((4,))]
    case = _stub_case({"y": 5.0}, storages, ((2,), (3,), (4,)), dist=(0, 2))

    InputGenerator(case)._override_tensors_from_attributes()

    np.testing.assert_array_equal(storages[1], np.full((3,), 5.0, np.float32))
    np.testing.assert_array_equal(storages[2], np.full((4,), 5.0, np.float32))
    assert "y" not in case.attributes


def test_tensorlist_per_tensor_values(monkeypatch):
    """TensorList 逐子张量对位：list 长度与子张量数一致。"""
    monkeypatch.setattr(aclnn_input_generate, "OpApiInfoKeeper", _fake_keeper(["x", "y"]))
    storages = [_rand((2,)), _rand((3,)), _rand((4,))]
    case = _stub_case({"y": [1.0, 2.0]}, storages, ((2,), (3,), (4,)), dist=(0, 2))

    InputGenerator(case)._override_tensors_from_attributes()

    np.testing.assert_array_equal(storages[1], np.full((3,), 1.0, np.float32))
    np.testing.assert_array_equal(storages[2], np.full((4,), 2.0, np.float32))


def test_tensorlist_length_mismatch_raises(monkeypatch):
    """TensorList 值长度与子张量数不一致 → ValueError。"""
    monkeypatch.setattr(aclnn_input_generate, "OpApiInfoKeeper", _fake_keeper(["x", "y"]))
    storages = [_rand((2,)), _rand((3,)), _rand((4,))]
    case = _stub_case({"y": [1.0, 2.0, 3.0]}, storages, ((2,), (3,), (4,)), dist=(0, 2))

    with pytest.raises(ValueError, match="length mismatch"):
        InputGenerator(case)._override_tensors_from_attributes()


def test_none_value_raises(monkeypatch):
    """张量形参名赋 None/空串 → 显式 ValueError。"""
    monkeypatch.setattr(aclnn_input_generate, "OpApiInfoKeeper", _fake_keeper(["x", "y"]))
    for bad in (None, "None", ""):
        storages = [_rand((2,)), _rand((4,))]
        case = _stub_case({"x": bad}, storages, ((2,), (4,)))
        with pytest.raises(ValueError, match="Invalid attributes"):
            InputGenerator(case)._override_tensors_from_attributes()


def test_dynamic_slot_skipped_with_warning(monkeypatch, caplog):
    """动态槽（无生成的 storage）告警跳过，键仍被消费（防 golden 残键毒化）。"""
    monkeypatch.setattr(aclnn_input_generate, "OpApiInfoKeeper", _fake_keeper(["x", "y"]))
    storages = [_rand((2,)), None]
    case = _stub_case({"y": 7.0}, storages, ((2,), None))

    with caplog.at_level("WARNING"):
        InputGenerator(case)._override_tensors_from_attributes()

    assert "not applied" in caplog.text
    assert "y" not in case.attributes


def test_noop_without_attributes_or_info(monkeypatch):
    """无 attributes / 未知 API → 空操作。"""
    monkeypatch.setattr(aclnn_input_generate, "OpApiInfoKeeper", _fake_keeper(["x", "y"]))
    storages = [_rand((2,)), _rand((4,))]
    case = _stub_case({}, storages, ((2,), (4,)))
    InputGenerator(case)._override_tensors_from_attributes()
    assert case.attributes == {}

    case = _stub_case({"x": 1.0}, storages, ((2,), (4,)))
    case.api_name = "aclnnUnknown"
    InputGenerator(case)._override_tensors_from_attributes()
    assert case.attributes == {"x": 1.0}
