#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Small, mode-neutral deterministic and batch-relation contracts."""

import logging
from collections.abc import Mapping
from numbers import Integral

BATCH_RELATION_FIELDS = ("batch_axis", "batch_slice_info", "batch_seed")


def has_complete_batch_relation(testcase):
    """Return whether all fields needed to construct a batch relation exist."""
    return all(getattr(testcase, name, None) is not None for name in BATCH_RELATION_FIELDS)


def batch_relation_kwargs(testcase):
    """Return all batch fields together, or no fields for a partial relation."""
    if not has_complete_batch_relation(testcase):
        return {}
    return {name: getattr(testcase, name) for name in BATCH_RELATION_FIELDS}


def parse_batch_relations(batch_axis, batch_slice_info, batch_seed, slot_idx=0):
    """relation 三字段嵌套结构 → (axes, 样本列表)；结构非法返回 None。

    嵌套契约（与 TestcaseBase._generate_batch_consistency_id 同构）：
    batch_axis[slot] = 逻辑轴元组；batch_slice_info[slot]/batch_seed[slot] =
    逐轴样本组，各组样本数一致；每样本 = 每轴一个 (start, stop, step) + seed。

    严格语义：字段缺失 / 轴组数不齐 / 类型非法 → None（消费方自行决定
    unsupported 或跳过）；单样本非法仅跳过该样本并 warning。样本切片只做
    结构校验，不做轴/区间语义校验（消费方按需二次校验）。
    """
    if batch_axis is None or batch_slice_info is None or batch_seed is None:
        return None
    axes = _slot_axes(batch_axis, slot_idx)
    if axes is None:
        return None
    groups = _slot_groups(batch_slice_info, batch_seed, slot_idx, len(axes))
    if groups is None:
        return None
    return axes, _collect_samples(axes, groups[0], groups[1])


def _slot_axes(batch_axis, slot_idx):
    """batch_axis[slot] → 轴号列表；结构非法返回 None。"""
    if not isinstance(batch_axis, (tuple, list)) or not batch_axis:
        return None
    try:
        axes = batch_axis[slot_idx]
    except (IndexError, TypeError):
        return None
    if not isinstance(axes, (tuple, list)) or not axes:
        return None
    parsed = []
    for axis in axes:
        if isinstance(axis, bool) or not isinstance(axis, int):
            return None
        parsed.append(int(axis))
    return parsed or None


def _slot_groups(batch_slice_info, batch_seed, slot_idx, axis_count):
    """slice/seed[slot] 逐轴样本组 → (axis_slices, axis_seeds)；非法返回 None。"""
    try:
        slice_groups = batch_slice_info[slot_idx]
        seed_groups = batch_seed[slot_idx]
    except (IndexError, TypeError):
        return None
    if slice_groups is None or seed_groups is None:
        return None
    if not isinstance(slice_groups, (tuple, list)) or not isinstance(seed_groups, (tuple, list)):
        return None
    if axis_count != len(slice_groups) or axis_count != len(seed_groups):
        return None
    sample_count = None
    axis_slices, axis_seeds = [], []
    for group_slices, group_seeds in zip(slice_groups, seed_groups):
        if (
            group_slices is None
            or group_seeds is None
            or not isinstance(group_slices, (tuple, list))
            or not isinstance(group_seeds, (tuple, list))
        ):
            return None
        if sample_count is None:
            sample_count = len(group_slices)
        if len(group_slices) != sample_count or len(group_seeds) != sample_count:
            return None
        axis_slices.append(group_slices)
        axis_seeds.append(group_seeds)
    return axis_slices, axis_seeds


def _collect_samples(axes, axis_slices, axis_seeds):
    """逐轴样本组 → 样本列表 [(每轴切片, 每轴 seed)]；单样本非法跳过。"""
    samples = []
    for index in range(len(axis_slices[0])):
        sample_slices, sample_seeds = [], []
        for axis_index in range(len(axes)):
            raw_slice = axis_slices[axis_index][index]
            raw_seed = axis_seeds[axis_index][index]
            if (
                not isinstance(raw_slice, (tuple, list))
                or len(raw_slice) != 3
                or not all(isinstance(item, int) and not isinstance(item, bool) for item in raw_slice)
                or isinstance(raw_seed, bool)
                or not isinstance(raw_seed, int)
            ):
                logging.warning(
                    "batch relation skip invalid sample: axes=%s slices=%r seed=%r",
                    axes,
                    raw_slice,
                    raw_seed,
                )
                break
            sample_slices.append(tuple(int(item) for item in raw_slice))
            sample_seeds.append(int(raw_seed))
        else:
            samples.append((sample_slices, sample_seeds))
    return samples


def resolve_deterministic_level(switches, testcase):
    """Resolve a case level while keeping a nonzero CLI level authoritative.

    ``batch_deterministic_level`` is an assets-private CSV attribute.  TTK only
    consumes it as an execution option when the command line did not select a
    nonzero level, so one input CSV can mix ordinary and deterministic cases.
    """
    cli_level = int(getattr(switches, "deterministic_level", 0) or 0)
    if cli_level:
        return cli_level

    attributes = getattr(testcase, "attributes", None)
    if not isinstance(attributes, Mapping):
        attributes = {}
    value = attributes.get("batch_deterministic_level")
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(
            f"{getattr(testcase, 'testcase_name', '<unknown>')}: "
            "attributes.batch_deterministic_level must be an integer from 0 through 3"
        )
    level = int(value)
    if level not in (0, 1, 2, 3):
        raise ValueError(
            f"{getattr(testcase, 'testcase_name', '<unknown>')}: "
            "attributes.batch_deterministic_level must be from 0 through 3"
        )
    return level
