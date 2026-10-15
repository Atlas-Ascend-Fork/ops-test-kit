# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for ttk.core_modules.common.testcase.infershape: broadcast/reduce/range."""

import pytest

from ttk.core_modules.common.testcase.infershape import (
    elewise,
    range_inference,
    reduce,
    shape_inference,
)


@pytest.mark.parametrize(
    ("shapes", "expected"),
    [
        (((8, 8), (8, 8)), ((8, 8),)),
        (((4, 1), (1, 8)), ((4, 8),)),
        (((8,), (4, 8)), ((4, 8),)),
        (((0,), (1,)), ((0,),)),
        (((5,), (1,)), ((5,),)),
        (((1,), (1,)), ((1,),)),
        (((-1, 3), (1, 3)), ((-1, 3),)),
        (((-1,), (-1,)), ((-1,),)),
        (((0, 2), (1,)), ((0, 2),)),
    ],
)
def test_elewise_broadcast(shapes, expected):
    assert elewise(*shapes) == expected


def test_elewise_incompatible_dims_raise():
    with pytest.raises(RuntimeError, match="Cannot broadcast incompatible shapes"):
        elewise((4, 8), (3, 8))


def test_elewise_multi_output_with_relations():
    out = elewise((4, 8), (4, 8), output_num=2, relations=((0,), (1,)))
    assert out == ((4, 8), (4, 8))


def test_elewise_no_input_shapes_raise():
    with pytest.raises(ValueError, match="no input shapes"):
        elewise(output_num=1)


def test_elewise_zero_output_num_raise():
    with pytest.raises(ValueError, match="no output shapes"):
        elewise((4,), output_num=0)


def test_elewise_invalid_shape_type():
    with pytest.raises(TypeError, match="invalid shapes"):
        elewise(("x", "y"))


@pytest.mark.parametrize(
    ("shape", "axes", "expected"),
    [
        ((2, 3, 4), (1,), (2, 1, 4)),
        ((2, 3, 4), None, (-1, -1, -1)),
        ((1, 3, 4), None, (1, -1, -1)),
    ],
)
def test_reduce(shape, axes, expected):
    assert reduce(shape, axes) == expected


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((4, 8), ((4, 4), (8, 8))),
        ((-1, 8, -2), ((1, None), (8, 8), (1, None))),
        (None, None),
    ],
)
def test_range_inference(shape, expected):
    assert range_inference(shape) == expected


@pytest.mark.parametrize(
    ("mode", "shapes", "args", "expected"),
    [
        ("ELEWISE", ((4, 8), (4, 8)), (1, None), ((4, 8),)),
        ("REDUCE", ((2, 3, 4),), ((1,), 1, None), ((2, 1, 4),)),
        ("RANGE", ((4, -1),), (None,), (((4, 4), (1, None)),)),
    ],
)
def test_shape_inference_dispatch(mode, shapes, args, expected):
    assert shape_inference(shapes, args, mode) == expected


def test_shape_inference_unknown_mode():
    with pytest.raises(RuntimeError, match="UNKNOWN MODE"):
        shape_inference(((4,),), (None,), "BOGUS")
