# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for ttk.core_modules.common.testcase.infershape: broadcast/reduce/range/transform."""

import math

import pytest

from ttk.core_modules.common.testcase.infershape import (
    elewise,
    range_inference,
    reduce,
    shape_inference,
    transform,
)


@pytest.mark.parametrize(
    ("shapes", "expected"),
    [
        (((8, 8), (8, 8)), ((8, 8),)),
        (((4, 1), (1, 8)), ((4, 8),)),
        (((8,), (4, 8)), ((4, 8),)),
        (((4, 8), (3, 8)), ((4, 8),)),
    ],
)
def test_elewise_broadcast(shapes, expected):
    assert elewise(*shapes) == expected


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


@pytest.mark.parametrize(
    ("shape", "cur", "target", "expected"),
    [
        ((1, 8, 8, 32), "NHWC", "NC1HWC0", (1, 2, 8, 8, 16)),
        ((1, 32, 8, 8), "NCHW", "NC1HWC0", (1, 2, 8, 8, 16)),
        ((1, 8, 32), "NWC", "NC1HWC0", (1, 2, 1, 8, 16)),
        ((1, 2, 8, 8, 32), "NDHWC", "NDC1HWC0", (1, 2, 2, 8, 8, 16)),
        ((2, 3), "NHWC", "FRACTAL_NZ", (1, 1, 16, 16)),
        ((1, 5, 2, 3), "NCHW", "FRACTAL_NZ", (1, 5, 1, 1, 16, 16)),
    ],
)
def test_transform_formats(shape, cur, target, expected):
    assert transform(shape, cur, target) == expected


def test_transform_unknown_source_raise():
    with pytest.raises(NotImplementedError, match="Cannot transform format"):
        transform((1, 2), "BOGUS", "NC1HWC0")


def test_transform_4d_infer_requires_4d():
    with pytest.raises(RuntimeError, match="4D to 5D"):
        transform((1, 2, 3), "NHWC", "NC1HWC0")


def test_transform_c_ceil_matches_math_ceil():
    c = 17
    assert transform((1, 8, 8, c), "NHWC", "NC1HWC0")[1] == math.ceil(c / 16)
