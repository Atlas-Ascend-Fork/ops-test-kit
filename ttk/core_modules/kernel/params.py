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
Operator parameter pickup: dict key adaptation to API signatures (axis/axes, _in__).

Kernel-only: CSV attributes → spec_tensors/spec_attrs 二分 + 算子函数签名键名适配。
"""


def adapt_pickup_by_names(params: dict, signatures: tuple) -> dict:
    """Pick up params matching signatures and adapt key names.

    For each param key, checks if it (or its transformed form) matches a
    signature name. Matching keys are renamed to the signature form.
    Non-matching keys are removed.

    Transformations checked (same order as pickup_by_names):
        1. axis/axes alias:  axis ↔ axes
        2. _in__ suffix:     param → param_in__
        3. _in__ strip:      param_in__ → param

    Args:
        params: Parameter dict (e.g. operator attributes).
        signatures: Tuple of function parameter names.

    Returns:
        Dict with keys renamed to match signatures.
    """
    params = params.copy()
    k_r = {"axis": "axes", "axes": "axis"}
    for param in tuple(params.keys()):
        if param in k_r and k_r[param] in signatures and k_r[param] not in params:
            params[k_r[param]] = params[param]
            del params[param]
            continue
        if param in k_r:
            k_r_with_tail = f"{k_r[param]}_in__"
            if k_r_with_tail in signatures and k_r_with_tail not in params:
                params[k_r_with_tail] = params[param]
                del params[param]
                continue
        param_with_tail = f"{param}_in__"
        if param_with_tail in signatures and param_with_tail not in params:
            params[param_with_tail] = params[param]
            del params[param]
            continue
        if param.endswith("_in__"):
            param_wo_tail = param[:-5]
            if param_wo_tail in signatures and param_wo_tail not in params:
                params[param_wo_tail] = params[param]
                del params[param]
                continue
        if param not in signatures:
            del params[param]
    return params


def pickup_by_names(attrs: dict, input_names) -> dict:
    """Extract attributes whose keys match input parameter names.

    Uses the same mapping logic as adapt_pickup_by_names: direct name match,
    _in__ suffix, and axis/axes alias. Keys keep their original form (no
    renaming).

    Args:
        attrs: Operator attributes dict (e.g. testcase.attributes).
        input_names: Sequence of input parameter names (e.g. from op_info["inputs"]).

    Returns:
        Dict of attributes that match input names. Caller can derive the
        remainder via `{k: v for k, v in attrs.items() if k not in result}`.
    """
    input_names = set(input_names)
    _alias = {"axis": "axes", "axes": "axis"}
    result = {}
    for key, value in attrs.items():
        if key in input_names:
            result[key] = value
        elif key.endswith("_in__"):
            base = key[:-5]
            if base in input_names or (base in _alias and _alias[base] in input_names):
                result[key] = value
        elif f"{key}_in__" in input_names or key in _alias and _alias[key] in input_names:
            result[key] = value
    return result
