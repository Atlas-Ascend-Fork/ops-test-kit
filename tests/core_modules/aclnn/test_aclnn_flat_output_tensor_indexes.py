# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for TestcaseAclnn.flat_output_tensor_indexes."""

from ttk.core_modules.aclnn.testcase import TestcaseAclnn


def _case(shapes, dtypes, output_tensor_indexes):
    case = TestcaseAclnn()
    case.tensor_view_shapes = shapes
    case.tensor_dtypes = dtypes
    case.output_tensor_indexes = tuple(output_tensor_indexes)
    return case


def test_list_plus_trailing_output_expands():
    """NonFinite 形态：列表槽(N=2) + 尾随单 tensor 槽 → 展平 3 个索引。"""
    case = _case(
        (((12984,), (12984,)), (32,)),
        (("float32", "float32"), "float32"),
        (0, 1),
    )
    assert case.tensor_list_dist == (2, 0)
    assert case.flat_output_tensor_indexes == (0, 1, 2)
    # flat 元数据族与 flat 索引同长（比对侧 idx 天然对齐的契约）
    assert len(case.flat_output_view_shapes) == len(case.flat_output_tensor_indexes)
    assert len(case.flat_output_storage_shapes) == len(case.flat_output_tensor_indexes)
    assert len(case.flat_output_view_strides) == len(case.flat_output_tensor_indexes)
    assert len(case.flat_output_view_offsets) == len(case.flat_output_tensor_indexes)
    assert len(case.flat_output_dtypes) == len(case.flat_output_tensor_indexes)
    assert case.flat_output_view_shapes == ((12984,), (12984,), (32,))


def test_pure_list_output_expands():
    """纯列表输出（ForeachAddScalar 形态）：单槽 N=3 → 3 个 flat 索引。"""
    case = _case(
        (((4,), (4,), (4,)),),
        (("float16", "float16", "float16"),),
        (0,),
    )
    assert case.tensor_list_dist == (3,)
    assert case.output_tensor_indexes == (0,)
    assert case.flat_output_tensor_indexes == (0, 1, 2)
    assert len(case.flat_output_view_shapes) == 3


def test_single_elem_list_aligns_with_slots():
    """elems=1：槽位数==张量数，flat 索引与槽序一一对应（旧行为不变）。"""
    case = _case(
        (((5,),), (2,)),
        (("float32",), "int32"),
        (0, 1),
    )
    assert case.tensor_list_dist == (1, 0)
    assert case.flat_output_tensor_indexes == (0, 1)
    assert len(case.flat_output_view_shapes) == 2


def test_plain_single_tensor_unchanged():
    """无 TensorList 分布：flat_output_tensor_indexes 即槽位索引（恒等）。"""
    case = _case(
        ((3,), (4,)),
        ("float32", "float32"),
        (1,),
    )
    assert case.tensor_list_dist == (0, 0)  # 无嵌套：每槽 0（非列表），max(d,1) 仍恒等
    assert case.flat_output_tensor_indexes == (1,)
    assert case.flat_output_view_shapes == ((4,),)


def test_leading_scalar_output_then_list():
    """普通槽在前、列表在后：flat 索引按前缀和正确交错（0=普通, 1..N=列表）。"""
    case = _case(
        ((32,), ((7,), (7,))),
        ("float32", ("float32", "float32")),
        (0, 1),
    )
    assert case.tensor_list_dist == (0, 2)
    assert case.flat_output_tensor_indexes == (0, 1, 2)
