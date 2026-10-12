#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""aclnn InputGenerator._apply_custom_input_result 单元测试。

覆盖：
- 返回新张量按输入槽位回写（写穿 storage），None 返回 = 原地写法兼容
- 纯输出槽不参与对位；None 槽位保持原值；TensorList 嵌套/平铺均可
- 数量不匹配、shape 不匹配、动态槽回值 → 显式 RuntimeError
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ttk.core_modules.aclnn.input_generate import InputGenerator


def _stub_case(flat_tensors, pure_outputs=()):
    return SimpleNamespace(
        testcase_name="t",
        flat_tensors=tuple(flat_tensors),
        pure_output_indexes=tuple(pure_outputs),
    )


def test_return_values_copied_into_views():
    """返回张量按槽位回写到既有视图对象。"""
    v0, v1 = torch.zeros(2, 3), torch.zeros(4)
    case = _stub_case([v0, v1])

    InputGenerator(case)._apply_custom_input_result((torch.full((2, 3), 1.0), torch.full((4,), 2.0)))

    assert torch.all(v0 == 1.0)
    assert torch.all(v1 == 2.0)


def test_return_write_through_to_backing_storage():
    """回写写穿 storage：from_numpy 共享内存，np_storages/dump 同源可见。"""
    storage = np.zeros((2, 3), np.float32)
    view = torch.from_numpy(storage)
    case = _stub_case([view])

    InputGenerator(case)._apply_custom_input_result([torch.full((2, 3), 5.0)])

    np.testing.assert_array_equal(storage, np.full((2, 3), 5.0, np.float32))


def test_none_return_is_inplace_noop():
    """整体返回 None = 原地写法：不回写、不报错。"""
    v0 = torch.zeros(3)
    case = _stub_case([v0])
    InputGenerator(case)._apply_custom_input_result(None)
    assert torch.all(v0 == 0.0)


def test_pure_output_slots_excluded_from_mapping():
    """返回值与输入槽对位（纯输出槽除外），输出视图不被改写。"""
    v_in, v_out = torch.zeros(2), torch.full((2,), 7.0)
    case = _stub_case([v_in, v_out], pure_outputs=(1,))

    InputGenerator(case)._apply_custom_input_result((torch.full((2,), 3.0),))

    assert torch.all(v_in == 3.0)
    assert torch.all(v_out == 7.0)  # 纯输出槽保持原值


def test_none_slot_value_keeps_original():
    """返回序列中的 None 槽位保持原值，其余照常回写。"""
    v0, v1 = torch.zeros(2), torch.zeros(2)
    case = _stub_case([v0, v1])

    InputGenerator(case)._apply_custom_input_result((None, torch.full((2,), 9.0)))

    assert torch.all(v0 == 0.0)
    assert torch.all(v1 == 9.0)


def test_tensorlist_nested_return_flattened():
    """TensorList 返回可嵌套可平铺：flatten 后与 flat 槽位对位。"""
    v0, v1, v2 = torch.zeros(2), torch.zeros(3), torch.zeros(4)
    case = _stub_case([v0, v1, v2])

    InputGenerator(case)._apply_custom_input_result(
        (torch.full((2,), 1.0), (torch.full((3,), 2.0), torch.full((4,), 3.0)))
    )

    assert torch.all(v0 == 1.0)
    assert torch.all(v1 == 2.0)
    assert torch.all(v2 == 3.0)


def test_bare_tensor_return_wrapped_as_single():
    """单张量裸返回（不包序列）视为单元素。"""
    v0 = torch.zeros(2)
    case = _stub_case([v0])
    InputGenerator(case)._apply_custom_input_result(torch.full((2,), 4.0))
    assert torch.all(v0 == 4.0)


def test_count_mismatch_raises():
    """返回数量与输入槽数不匹配 → RuntimeError（防止错位回写）。"""
    v0, v1 = torch.zeros(2), torch.zeros(2)
    case = _stub_case([v0, v1])
    with pytest.raises(RuntimeError, match="returned 1 tensors, expected 2 input tensor slots"):
        InputGenerator(case)._apply_custom_input_result((torch.full((2,), 1.0),))


def test_shape_mismatch_raises():
    """返回张量 shape 与槽位不符 → RuntimeError（带槽位上下文）。"""
    v0 = torch.zeros(2, 3)
    case = _stub_case([v0])
    with pytest.raises(RuntimeError, match="flat tensor slot 0"):
        InputGenerator(case)._apply_custom_input_result((torch.zeros(4, 5),))


def test_dynamic_slot_with_value_raises_and_none_skips():
    """动态槽（无生成张量）回值 → RuntimeError；None 值 → 跳过。"""
    v0 = torch.zeros(2)
    case = _stub_case([v0, None])
    with pytest.raises(RuntimeError, match="no generated tensor"):
        InputGenerator(case)._apply_custom_input_result((torch.full((2,), 1.0), torch.zeros(3)))

    InputGenerator(case)._apply_custom_input_result((torch.full((2,), 1.0), None))
    assert torch.all(v0 == 1.0)


def test_numpy_view_write():
    """use_numpy 模式（视图为 ndarray）：view[...] = val 回写。"""
    storage = np.zeros((3,), np.float32)
    case = _stub_case([storage])

    InputGenerator(case)._apply_custom_input_result([np.full((3,), 6.0, np.float32)])

    np.testing.assert_array_equal(storage, np.full((3,), 6.0, np.float32))
