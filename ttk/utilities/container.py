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
Container / TensorList nesting protocol utilities
"""

# Standard Packages
import logging
from typing import Sequence


def get(container: Sequence, idx: int, out_of_range=None):
    """
    If a container contains only one element, return that element for whatever the idx is
    :param container: A container
    :param idx: index
    :param out_of_range: fallback value returned on out-of-range access; when not
        given, an error is logged (idx/length/container) and IndexError propagates
    :return: element
    """
    if len(container) == 1:
        return container[-1]
    if idx >= len(container) or (idx < 0 and abs(idx) > len(container)):
        if out_of_range is not None:
            return out_of_range
        logging.error(
            "Container out-of-range access: idx=%d, len=%d, container=%s; "
            "no out_of_range fallback given, IndexError will propagate",
            idx,
            len(container),
            str(container),
        )
    return container[idx]


def deep_flatten(sequence: Sequence) -> tuple:
    """Recursively flatten all tuple/list/set nesting.

    For atomic elements (ndarray, str, scalar): the element itself is never
    traversed into. Only tuple/list/set containers are recursively expanded.

    ((arr1, arr2), arr3) → (arr1, arr2, arr3)
    (('f32', 'f32'), 'int32') → ('f32', 'f32', 'int32')
    (arr1, arr2) → (arr1, arr2)  # already flat
    """
    result = []
    for ele in sequence:
        if isinstance(ele, (tuple, list, set)):
            for true_ele in deep_flatten(ele):
                result.append(true_ele)
        else:
            result.append(ele)
    return tuple(result)


def _is_tensor_list_group(element) -> bool:
    """A slot is a TensorList group when ANY sub-element is a shape/range
    sequence; None slots are legal at any position (incl. leading), so the
    first element alone must not decide. All-None or empty groups stay
    ambiguous and are treated as leaves.
    """
    return isinstance(element, (tuple, list)) and any(isinstance(sub, (tuple, list)) for sub in element)


def flatten_groups(nested) -> tuple:
    """Flatten one level of TensorList grouping, preserving leaf tuples.

    Unlike deep_flatten which recursively flattens everything, this preserves
    leaf tuples (like (3,3)) — a top element that contains any container
    sub-element is a group and gets expanded; otherwise it is a leaf
    (shape/range/None) and is kept as-is. Designed for shape/range
    declaration columns.

    (((3,3),(3,2)),(3,5)) → ((3,3),(3,2),(3,5))
    ((8,16),(32,)) → ((8,16),(32,))  # no nested groups, preserved
    None values are preserved, including None slots inside groups.
    """
    flat = []
    for element in nested:
        if element is None:
            flat.append(None)
        elif _is_tensor_list_group(element):
            flat.extend(element)
        else:
            flat.append(element)
    return tuple(flat)


def infer_list_distribution_from_nesting(nested):
    """Infer tensor_list_distribution from nested structure.

    Each top-level element:
      - tuple/list containing shape/range sub-sequences → TensorList, count = len
      - single shape (tuple-of-ints) or None → single tensor, count = 0

    Example: (((3,3),(3,2)),(3,5)) → (2, 0)
    """
    distribution = []
    for element in nested:
        if element is None:
            distribution.append(0)
        elif _is_tensor_list_group(element):
            distribution.append(len(element))
        else:
            distribution.append(0)
    return tuple(distribution)


def flatten_by_distribution(values, distribution):
    """Flatten a distribution-aligned field to one value per tensor.

    Inverse of nest_by_distribution. Accepted input conventions (in order):
      - whole-column broadcast: a bare scalar (or one-element sequence) is
        broadcast to every slot
      - slot-aligned: one value per slot (len == len(distribution))
      - pre-flattened: one value per tensor (len == sum(max(d, 1)));
        folded back to slots via nest_by_distribution
      - anything else raises ValueError instead of silently mis-mapping

    Within a TensorList slot, a scalar (or one-element sequence) is broadcast
    to every tensor of that slot; a full-length group is consumed as-is.
    """
    if not isinstance(values, (tuple, list)):
        values = (values,)
    dist = tuple(distribution or ())
    if len(values) == 1 and len(dist) > 1:
        values = tuple(values) * len(dist)
    elif len(values) != len(dist):
        flat_count = sum(max(d, 1) for d in dist)
        if len(values) == flat_count:
            values = nest_by_distribution(values, dist)
        else:
            raise ValueError(
                f"flatten_by_distribution: values {values!r} ({len(values)}) not aligned "
                f"with distribution {dist!r} ({len(dist)} slots, {flat_count} tensors)"
            )
    result = []
    for vi, num in enumerate(dist):
        val = values[vi]
        if num > 0:
            if isinstance(val, (tuple, list)):
                if len(val) == num:
                    result.extend(val)
                elif len(val) == 1:
                    result.extend([val[0]] * num)
                else:
                    raise ValueError(
                        f"flatten_by_distribution: slot {vi} expects {num} values, got {len(val)}: {val!r}"
                    )
            else:
                result.extend([val] * num)
        else:
            result.append(val)
    return tuple(result)


def split_distribution(dist, first_count):
    """Split a whole-layout slot table at the flat boundary `first_count`.

    A whole-layout dist covers input and output slots concatenated; folding
    consumes num>0 elements for a group slot and 1 for a single slot. Returns
    (first_slots, rest_slots) such that slots never straddle the boundary —
    mirrors the folding consumption formula exactly.
    """
    consumed = 0
    for boundary, num in enumerate(dist):
        if consumed >= first_count:
            return tuple(dist[:boundary]), tuple(dist[boundary:])
        consumed += num if num > 0 else 1
    return tuple(dist), ()


def nest_by_distribution(inputs: Sequence, distribution: Sequence):
    """Fold a flat sequence back into nested form, slot by slot.

    Exact inverse of flatten_by_distribution. A dist of 0 consumes one
    element as a single slot; dist of n consumes n elements as one group
    slot (the slice type follows the input sequence). Requires dist aligned
    with inputs: dist longer than inputs is a caller bug and raises
    IndexError. Trailing elements beyond dist are appended as single slots.
    """
    if distribution:
        result = []
        last_num = 0
        for num in distribution:
            if num == 0:
                result.append(inputs[last_num])
                last_num += 1
            else:
                result.append(inputs[last_num : last_num + num])
                last_num += num
        if last_num < len(inputs):
            result.extend(inputs[last_num:])
    else:
        result = inputs
    return result
