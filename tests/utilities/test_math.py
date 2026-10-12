# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for ttk.utilities.math: integer alignment algebra and bit-exact +/-0 predicates."""

import pytest

from ttk.utilities.math import align, ceil_div, is_negative_zero, is_positive_zero, lcm


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (7, 2, 4),
        (8, 2, 4),
        (0, 3, 0),
        (1, 16, 1),
    ],
    ids=["rounds-up", "exact", "zero", "small"],
)
def test_ceil_div(a, b, expected):
    assert ceil_div(a, b) == expected


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (17, 16, 32),
        (32, 16, 32),
        (0, 16, 0),
    ],
    ids=["rounds-up", "already-aligned", "zero"],
)
def test_align(a, b, expected):
    assert align(a, b) == expected


def test_lcm():
    assert lcm(4, 6) == 12
    assert lcm(5, 5) == 5


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-0.0, True),
        (0.0, False),
        (0, False),
        (-1.0, False),
    ],
    ids=["neg-zero", "pos-zero", "int-zero", "nonzero"],
)
def test_is_negative_zero(value, expected):
    assert is_negative_zero(value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, True),
        (-0.0, False),
        (0, True),
        (1.5, False),
    ],
    ids=["pos-zero", "neg-zero", "int-zero", "nonzero"],
)
def test_is_positive_zero(value, expected):
    assert is_positive_zero(value) is expected
