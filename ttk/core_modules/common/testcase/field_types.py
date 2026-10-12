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
CSV Field types
"""

__all__ = ["FIELD_TYPES"]

# Standard Packages
from enum import Enum, auto


class FIELD_TYPES(Enum):
    """
    Expected types for all columns
    """

    STRING = auto()  # This is a pure string
    SHAPELIKE_DYN = auto()  # This is a dynamic shape, which means it supports negative values
    SHAPELIKE_STC = auto()  # This is a static shape, which means it doesn't support negative values
    SHAPELIKE_DYN_EX = auto()  # This is a dynamic output shape, which means it supports inference repr
    SHAPELIKE_STC_EX = auto()  # This is a static output shape, which means it supports inference repr
    SHAPELIKE_FLOAT = auto()  # This is a float shape like object, which means it supports float dims
    SHAPELIKE_FLOAT_SIGNED = auto()  # This is a signed float shape like object, which means it supports negative
    STRING_CONTAINER = auto()  # This is a string container, which means it must be a tuple[str]
    INT = auto()  # This is an integer, which means it must be an int
    INT_CONTAINER = auto()
    BOOL = auto()  # This is a pure bool
    DICT = auto()  # This is a pure dict
    FREE_EVAL = auto()  # FREE Evaluation
    SHAPE_STRIDE = auto()  # shape stride
    SHAPELIKE_STC_NESTED = auto()  # static shape with TensorList nesting support
    SHAPELIKE_STC_EX_NESTED = auto()  # static shape with TensorList nesting + inference string support
    STRING_SCALAR_NESTED = auto()  # string scalar with TensorList nesting support
    FLOAT_SCALAR_NESTED = auto()  # float/int scalar with TensorList nesting support
    INT_CONTAINER_NESTED = auto()  # int container with TensorList nesting support
    SHAPELIKE_FLOAT_SIGNED_NESTED = auto()  # signed float shapelike with TensorList nesting support
    SHAPELIKE_FLOAT_NESTED = auto()  # float shapelike with TensorList nesting support (positive only)
    SHAPELIKE_DYN_NESTED = auto()  # dynamic shape with TensorList nesting support, allows -1/-2
