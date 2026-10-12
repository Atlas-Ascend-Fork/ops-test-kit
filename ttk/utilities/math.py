#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Integer alignment algebra and bit-exact +/-0 predicates.
"""

__all__ = [
    "ceil_div",
    "align",
    "lcm",
    "is_negative_zero",
    "is_positive_zero",
]


# Standard Packages
import numpy


def ceil_div(a, b):
    """Ceiling of a / b for non-negative ints."""
    return (a + b - 1) // b


def align(a, b):
    """Round a up to the next multiple of b."""
    return ceil_div(a, b) * b


def lcm(a, b):
    """least common multiple of a and b"""
    return numpy.lcm(a, b)


def is_negative_zero(number: float) -> bool:
    # torch.signbit behavior varies across torch versions, so numpy.signbit is used here
    return bool(number == 0 and numpy.signbit(number))


def is_positive_zero(number: float) -> bool:
    return bool(number == 0 and not numpy.signbit(number))
