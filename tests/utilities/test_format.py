# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for ttk.utilities.format: NPU format predicates, shape algebra and
ND <-> private-format transforms (roundtrip where an inverse exists, shape
pin otherwise)."""

import numpy as np
import pytest

from ttk.utilities.format import (
    align_factor,
    from_fractal_nz,
    from_fractal_z_3d,
    from_nc1hwc0,
    from_ndc1hwc0,
    is_nchw_like,
    is_ndhwc_like,
    is_transformable,
    nd_shape_to_5hd,
    nd_shape_to_nz,
    nd_to_fractal_nz,
    nd_to_fractal_z,
    resolve_c0,
    to_fractal_z,
    to_fractal_z_3d,
    to_fractal_z_c04,
    to_nc1hwc0,
    to_ndc1hwc0,
    transform,
    transpose_axes,
)


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        ("float16", 16),
        ("int8", 32),
        ("int64", 4),
        ("bogus", 16),
    ],
)
def test_align_factor(dtype, expected):
    assert align_factor(dtype) == expected


@pytest.mark.parametrize(
    ("dtype", "target", "expected"),
    [
        ("float16", [1, 2, 8, 8, 24], 24),
        ("int8", None, 32),
        pytest.param(np.dtype("float16"), None, 16, id="numpy-dtype"),
    ],
    ids=["target-shape", "fallback", "numpy-dtype"],
)
def test_resolve_c0(dtype, target, expected):
    assert resolve_c0(dtype, target) == expected


@pytest.mark.parametrize(
    ("shape", "fmt", "expected"),
    [
        ((1, 32, 8, 8), "NCHW", True),
        ((1, 32, 8), "NCHW", False),
        ((1, 32, 8, 8), "NCHX", False),
    ],
    ids=["ok", "bad-rank", "bad-format"],
)
def test_is_nchw_like(shape, fmt, expected):
    assert is_nchw_like(shape, fmt) is expected


@pytest.mark.parametrize(
    ("shape", "fmt", "expected"),
    [
        ((1, 2, 32, 8, 8), "NCDHW", True),
        ((1, 32, 8, 8), "NCDHW", False),
        ((1, 2, 32, 8, 8), "NDHWW", False),
    ],
    ids=["ok", "bad-rank", "bad-format"],
)
def test_is_ndhwc_like(shape, fmt, expected):
    assert is_ndhwc_like(shape, fmt) is expected


@pytest.mark.parametrize(
    ("shape", "fmt", "expected"),
    [
        ((1, 32, 8, 8), "NCHW", (1, 2, 8, 8, 16)),
        ((1, 8, 8, 32), "NHWC", (1, 2, 8, 8, 16)),
    ],
)
def test_nd_shape_to_5hd(shape, fmt, expected):
    assert nd_shape_to_5hd(shape, fmt) == expected


def test_nd_shape_to_nz():
    assert nd_shape_to_nz((2, 3)) == (1, 1, 16, 16)


def test_nd_shape_to_5hd_rejects_non_nchw():
    with pytest.raises(RuntimeError, match="not NCHW-like"):
        nd_shape_to_5hd((1, 2, 3), "NCH")


def test_nchw_to_nc1hwc0_and_back():
    data = np.arange(1 * 32 * 8 * 8, dtype=np.float16).reshape(1, 32, 8, 8)
    fractal = to_nc1hwc0(data, "NCHW")
    assert fractal.shape == (1, 2, 8, 8, 16)
    np.testing.assert_array_equal(from_nc1hwc0(fractal, (1, 32, 8, 8), "NCHW"), data)


def test_ndchw_to_ndc1hwc0_and_back():
    # c=20 > c0=16: exercises main-block + tail-block concat path
    data = np.arange(1 * 20 * 2 * 4 * 5, dtype=np.float16).reshape(1, 20, 2, 4, 5)
    fractal = to_ndc1hwc0(data, "NCDHW")
    assert fractal.shape == (1, 2, 2, 4, 5, 16)
    np.testing.assert_array_equal(from_ndc1hwc0(fractal, (1, 20, 2, 4, 5), "NCDHW"), data)


def test_nd_to_fractal_nz_and_back():
    data = np.arange(2 * 3, dtype=np.float16).reshape(2, 3)
    fractal = nd_to_fractal_nz(data)
    assert fractal.shape == (1, 1, 16, 16)
    np.testing.assert_array_equal(from_fractal_nz(fractal, (2, 3)), data)


def test_nd_to_fractal_nz_4d_and_back():
    data = np.arange(2 * 5 * 4 * 7, dtype=np.float16).reshape(2, 5, 4, 7)
    fractal = nd_to_fractal_nz(data)
    np.testing.assert_array_equal(from_fractal_nz(fractal, (2, 5, 4, 7)), data)


def test_to_fractal_z_shape():
    # cout=2, cin=8, 3x3: h*w*cin1=9 blocks, cout1=1, (16, c0=16)
    weight = np.zeros((2, 8, 3, 3), dtype=np.float16)
    assert to_fractal_z(weight, "NCHW").shape == (9, 1, 16, 16)


def test_to_fractal_z_c04_shape():
    # GE format_transfer_fz_c04: (ceil(4*h*w/c0), ceil(n/16), 16, c0)
    weight = np.zeros((2, 4, 3, 3), dtype=np.float16)
    assert to_fractal_z_c04(weight, "NCHW").shape == (3, 1, 16, 16)


def test_to_fractal_z_c04_rejects_c_above_4():
    # 原用例输入 (2,8,3,3)：GE 语义下 FRACTAL_Z_C04 要求 C<=4（原先 IndexError，现显式拒绝）
    weight = np.zeros((2, 8, 3, 3), dtype=np.float16)
    with pytest.raises(RuntimeError, match="C <= 4"):
        to_fractal_z_c04(weight, "NCHW")


def test_nd_to_fractal_z_shape():
    data = np.zeros((2, 5, 4, 7), dtype=np.float16)
    assert nd_to_fractal_z(data).shape == (2, 5, 1, 1, 16, 16)


def test_to_fractal_z_3d_and_back():
    weight = np.arange(1 * 4 * 2 * 3 * 3, dtype=np.float16).reshape(1, 4, 2, 3, 3)
    fractal = to_fractal_z_3d(weight, "NCDHW")
    assert fractal.shape == (18, 1, 16, 16)
    np.testing.assert_array_equal(from_fractal_z_3d(fractal, list(weight.shape), "NCDHW"), weight)


@pytest.mark.parametrize(
    ("ori", "target", "expected"),
    [
        ("NCHW", "NC1HWC0", True),
        ("ND", "FRACTAL_NZ", True),
        ("BOGUS", "NC1HWC0", False),
        ("NCHW", "BOGUS", False),
    ],
)
def test_is_transformable(ori, target, expected):
    assert is_transformable(ori, target) is expected


def test_transform_returns_none_when_not_transformable():
    assert transform(np.zeros((1, 2, 3, 4), dtype=np.float16), "BOGUS", "NC1HWC0") is None


def test_transform_dispatch_roundtrip():
    # reverse dispatch: from_nc1hwc0 has no ori_format param — exercises kwargs skip
    data = np.arange(1 * 16 * 8 * 8, dtype=np.float16).reshape(1, 16, 8, 8)
    fractal = transform(data, "NCHW", "NC1HWC0")
    assert fractal.shape == (1, 1, 8, 8, 16)
    np.testing.assert_array_equal(transform(fractal, "NC1HWC0", "NCHW", target_shape=[1, 16, 8, 8]), data)


def test_transpose_axes():
    assert transpose_axes(2, [2, 0, 1, 3]) == [0, 1, 4, 2, 3, 5]
