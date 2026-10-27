#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""batch 片段替换：框架侧逐样本区段重生成。

- apply_slot_batch_replacement：单槽指令重放——kernel/GEIR 在 ori 数组生成后调用
  （transform 前，逻辑布局无歧义）；aclnn/e2e 由 generate_np_storages 在整张量
  生成后调用（view/storage 同构防护见 input_data._is_view_storage_isomorphic）
- apply_batch_replacement：全槽遍历（备用整批入口）

一致性契约：同 (sample_seed, 区段 shape, dtype, data_range, distribution) →
逐位一致。RandomData 消费全局 numpy.random 流，播种用 get_state/set_state
包裹（方案 A），替换后全局随机态零漂移。batch_axis 语义 = 用户声明的逻辑
shape 上的轴（aclnn/e2e 视图 shape / kernel-GEIR ori shape）。
"""

__all__ = ["apply_batch_replacement", "apply_slot_batch_replacement"]

# Standard Packages
import logging

# Third-Party Packages
import numpy

from ....utilities import RandomData, get
from .batch import has_complete_batch_relation, parse_batch_relations


def _slot_instructions(testcase, slot_idx):
    """槽位替换指令迭代器：(axis_idx, (start, stop, step), sample_seed)。

    嵌套结构与 bcid 派生（TestcaseBase._generate_batch_consistency_id）
    同构：slot -> 轴组 -> 样本，经 parse_batch_relations 解析（单样本非法
    跳过；结构不完整 → 无指令）。未声明的槽（get 越界 → None）无指令。
    """
    relations = parse_batch_relations(
        getattr(testcase, "batch_axis", None),
        getattr(testcase, "batch_slice_info", None),
        getattr(testcase, "batch_seed", None),
        slot_idx,
    )
    if relations is None:
        return
    axes, samples = relations
    for axis_index, axis_idx in enumerate(axes):
        for sample_slices, sample_seeds in samples:
            yield axis_idx, sample_slices[axis_index], sample_seeds[axis_index]


def apply_slot_batch_replacement(testcase, slot_idx, arr, *, dtype, data_range, distribution):
    """单槽逐样本区段重生成（原地写 arr）。

    尽力而为语义（对齐 slice_batch_hashes）：relation 不完整 / 指令非法 /
    生成异常均 warning 跳过，不抛不置 invalid——batch 一致性是辅助校验，
    不阻断主流程。多轴组重叠区后写覆盖先写。
    """
    if arr is None or not has_complete_batch_relation(testcase):
        return
    for axis_idx, sample_slice, sample_seed in _slot_instructions(testcase, slot_idx):
        try:
            start, stop, step = sample_slice[0], sample_slice[1], sample_slice[2]
            length = len(range(start, stop, step))
            axis_ok = isinstance(axis_idx, int) and not isinstance(axis_idx, bool) and 0 <= axis_idx < arr.ndim
            if not axis_ok or length == 0:
                logging.warning(
                    "[%s] batch replacement skip slot %d: invalid axis=%r slice=%r",
                    getattr(testcase, "testcase_name", None),
                    slot_idx,
                    axis_idx,
                    sample_slice,
                )
                continue
            region_shape = arr.shape[:axis_idx] + (length,) + arr.shape[axis_idx + 1 :]
            state = numpy.random.get_state()
            try:
                numpy.random.seed(int(sample_seed))
                region = RandomData(dtype, region_shape, data_range).generate(distribution)
            finally:
                numpy.random.set_state(state)
            indexer = (slice(None),) * axis_idx + (slice(start, stop, step),)
            arr[indexer] = region
        except Exception as exc:  # 替换尽力而为，不阻断主流程
            logging.warning(
                "[%s] batch replacement skip slot %d instruction (axis=%r slice=%r): %s",
                getattr(testcase, "testcase_name", None),
                slot_idx,
                axis_idx,
                sample_slice,
                exc,
            )
            continue


def apply_batch_replacement(testcase, flat_arrays, *, flat_dtypes, flat_ranges, distribution):
    """全槽替换——aclnn/e2e 在整张量生成后调用（generate_np_storages）。"""
    for slot_idx, arr in enumerate(flat_arrays):
        if arr is None:
            continue
        apply_slot_batch_replacement(
            testcase,
            slot_idx,
            arr,
            dtype=get(flat_dtypes, slot_idx),
            data_range=get(flat_ranges, slot_idx),
            distribution=distribution,
        )
