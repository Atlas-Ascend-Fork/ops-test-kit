#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shape algebra: element count / span / strides / validation."""

from math import prod
from typing import Sequence, Tuple, Union


def shape_numel(shape: Tuple[int, ...]) -> int:
    """Number of elements = product of dims; empty shape () → 1 (0-d scalar)."""
    return prod(shape)


def shape_span(shape: Union[list, tuple], strides: Union[list, tuple]) -> int:
    """Flat-element span of a strided view: 1 + Σ (dim-1)*stride; 0 if any dim is 0.

    NOT numel: the span of a non-contiguous view (e.g. transposed) is how many
    flat elements the (shape, strides) access touches — the buffer size an
    as_strided-style view must reserve. Bounds check: span + offset <= numel(storage).
    """
    span = 1
    for dim, stride in zip(shape, strides):
        if dim == 0:
            return 0
        span += (dim - 1) * stride
    return span


def shape_stride(shape: Union[list, tuple]):
    """Row-major contiguous element strides: (d0..dn) → (Πd[1:], ..., d[n-1], 1)."""
    strides = [1] * len(shape)
    for idx in range(len(shape) - 2, -1, -1):
        strides[idx] = strides[idx + 1] * shape[idx + 1]
    return tuple(strides)


def is_shape(candidate, allowed_type: tuple = (int,)) -> bool:
    """True if candidate is a Sequence whose every dim is an instance of allowed_type."""
    if not isinstance(candidate, Sequence):
        return False
    return all(isinstance(dim, allowed_type) for dim in candidate)


def promote_scalar_shapes(shapes) -> tuple:
    """Lift empty shape declarations () to (1,) — a 0-d scalar occupies one element.

    None entries are preserved; already-shaped entries pass through unchanged.
    """
    if shapes is None:
        return ()
    return tuple(None if s is None else s if len(s) > 0 else (1,) for s in shapes)
