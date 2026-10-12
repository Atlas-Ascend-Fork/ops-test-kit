# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""shape 测试：形状代数——numel/span/stride 语义与 shape 校验/标量升维。"""

import pytest

from ttk.utilities.shape import (
    is_shape,
    promote_scalar_shapes,
    shape_numel,
    shape_span,
    shape_stride,
)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((2, 3, 4), 24),
        ((), 1),  # 0-d 标量占 1 元素
        ((2, 0, 3), 0),
    ],
)
def test_shape_numel(shape, expected):
    assert shape_numel(shape) == expected


def test_shape_stride():
    assert shape_stride((2, 3, 4)) == (12, 4, 1)
    assert shape_stride(()) == ()
    assert shape_stride((5,)) == (1,)


@pytest.mark.parametrize(
    ("shape", "strides", "expected"),
    [
        ((2, 3), (4, 1), 7),  # (2-1)*4 + (3-1)*1 + 1：非连续 span > numel(6)
        ((2, 3), (1, 2), 6),  # 列主序：span == numel
        ((2, 0), (1, 1), 0),  # 任一维为 0 → 0
        ((), (), 1),  # 空视图 span=1
    ],
)
def test_shape_span_not_numel(shape, strides, expected):
    """span 语义：strided view 触及的最大扁平元素数（缓冲区大小），非 numel。"""
    assert shape_span(shape, strides) == expected


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        ((2, 3), True),
        ([2, 3], True),
        ((2, -1), True),
        ((2, 3.0), False),
        ("shape", False),
        (2, False),
    ],
)
def test_is_shape(candidate, expected):
    assert is_shape(candidate) is expected


def test_promote_scalar_shapes():
    """空 shape () → (1,)（标量升一维），None 原样保留。"""
    assert promote_scalar_shapes(((), (2, 3), None)) == ((1,), (2, 3), None)
    assert promote_scalar_shapes(None) == ()
