#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""batch 一致性（哈希回传替代全量驻留）。

设计：
- worker 比对后：assemble_batch_summary 按 slot 0 首轴组的逻辑轴切 output 0，
  逐样本 MD5 → batch_summary（随 ReturnStructure 回传）——主进程内存
  O(用例×样本×32B)，替代旧"全量 output_bytes 回传驻留"
- 主进程 cleanup：compare_batch_consistency 按 batch_consistency_id 分组 →
  同关系位置 hash 集合基数判定（比较器不触数据本体，逻辑重画为"组装"）

接线现状：四通路 worker 侧 assemble（geir/kernel/aclnn/e2e）+ 主进程
InstanceBase.cleanup 侧组装比对。
"""

__all__ = ["slice_batch_hashes", "assemble_batch_summary", "compare_batch_consistency", "BatchSummary"]

# Standard Packages
import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_SLICE_UNSET = "UNSLICEABLE"
_MISSING_META = "BATCH_META_MISSING"
_HASH_LEN = 32  # md5 hexdigest 截断长度（每样本 32B 十六进制）


@dataclass
class BatchSummary:
    """单用例的 batch 摘要（随 ReturnStructure 回传；主进程内存受控）。"""

    testcase_name: str
    batch_consistency_id: str
    sample_hashes: Optional[List[str]] = None  # None = 不支持切片（记录原因）
    unsupported_reason: Optional[str] = None
    batch_axis: Optional[int] = None
    batch_slice_info: Optional[Any] = None

    def as_dict(self) -> dict:
        return {
            "testcase": self.testcase_name,
            "sample_hashes": self.sample_hashes,
            "unsupported_reason": self.unsupported_reason,
            "batch_axis": self.batch_axis,
            "batch_slice_info": _slice_repr(self.batch_slice_info),
        }


def _slice_repr(info) -> Optional[str]:
    if info is None:
        return None
    return str(info)


def slice_batch_hashes(
    output: Any,
    testcase_name: str,
    batch_consistency_id: str,
    batch_axis: Optional[int],
    batch_slice_info: Optional[Any],
) -> BatchSummary:
    """worker 比对后切片哈希：按 batch_axis 切 output 0，逐样本 MD5。

    非 ndarray 的 array-like（如 aclnn 比对后的 torch 视图）先 asarray 归一。
    切片失败（维度不足/元数据缺失）记 unsupported_reason，不抛——
    batch 一致性是辅助校验，不阻断主流程。"""
    summary = BatchSummary(
        testcase_name=testcase_name,
        batch_consistency_id=batch_consistency_id,
        batch_axis=batch_axis,
        batch_slice_info=batch_slice_info,
    )
    if batch_axis is None or batch_slice_info is None:
        summary.unsupported_reason = _MISSING_META
        return summary
    if not isinstance(output, np.ndarray):
        try:
            output = np.asarray(output)
        except Exception:  # noqa: BLE102 归一尽力而为
            summary.unsupported_reason = _SLICE_UNSET
            return summary
    if output.ndim <= batch_axis:
        summary.unsupported_reason = _SLICE_UNSET
        return summary
    try:
        summary.sample_hashes = []
        for sample in output:
            digest = hashlib.md5(np.ascontiguousarray(sample).tobytes()).hexdigest()[:_HASH_LEN]  # noqa: S324
            summary.sample_hashes.append(digest)
    except Exception as exc:  # noqa: BLE102 切片尽力而为
        summary.sample_hashes = None
        summary.unsupported_reason = f"{_SLICE_UNSET}: {exc}"
    return summary


def _output_batch_axis(testcase) -> Optional[int]:
    """输出切片轴：slot 0 首轴组的逻辑轴（FA Q-only 契约，与
    scripts/compare_batch_consistency.py 的 Relation 推导同构）。

    batch_axis 是嵌套结构（输入槽 → 轴组 → 逻辑轴）；宽容接受标量
    （非 FREE_EVAL 形态的调用方，如单测/插件直构）。
    """
    axes = getattr(testcase, "batch_axis", None)
    if axes is None:
        return None
    if isinstance(axes, int) and not isinstance(axes, bool):
        return axes
    if not isinstance(axes, (tuple, list)) or not axes:
        return None
    group = axes[0]
    if not isinstance(group, (tuple, list)) or not group:
        return None
    axis = group[0]
    if isinstance(axis, bool) or not isinstance(axis, int):
        return None
    return axis


def assemble_batch_summary(
    outputs: Sequence,
    testcase,
    batch_consistency_id: Optional[str],
) -> Optional[BatchSummary]:
    """result construct 的装配入口：id 为 None（不参与分组）或输出空 → None。

    输出轴不可派生（batch_axis 空 / 首组非整型）时产出带 unsupported_reason
    的 summary（参与分组但跳过比对），而非静默 None。"""
    if batch_consistency_id is None or not outputs:
        return None
    output = outputs[0]
    if isinstance(output, (str, bytes)) or output is None:
        return None
    axis = _output_batch_axis(testcase)
    if axis is None:
        summary = BatchSummary(
            testcase_name=getattr(testcase, "testcase_name", None),
            batch_consistency_id=batch_consistency_id,
            batch_axis=None,
            batch_slice_info=getattr(testcase, "batch_slice_info", None),
        )
        summary.unsupported_reason = _MISSING_META
        return summary
    return slice_batch_hashes(
        output,
        getattr(testcase, "testcase_name", None),
        batch_consistency_id,
        axis,
        getattr(testcase, "batch_slice_info", None),
    )


def compare_batch_consistency(summaries: Dict[str, BatchSummary]) -> Tuple[bool, Dict[str, List[str]]]:
    """cleanup 组装比对：按 id 分组 → 同关系位置 hash 集合基数判定。

    summaries: {testcase_name: BatchSummary}（值自带 batch_consistency_id，
    同 id 多用例归入同组比对——键是 testcase 名而非 id，同 id 的多用例
    才能共存于 dict）。
    判定：同 id 组内，所有样本数一致的用例在每个位置 hash 集合基数 == 1 → 一致。
    返回 (all_consistent, {id: 不一致位置描述})；unsupported 用例跳过并 log。
    """
    groups: Dict[str, List[BatchSummary]] = {}
    for summary in summaries.values():
        if summary.sample_hashes is None:
            logging.info("batch consistency skip [%s]: %s", summary.testcase_name, summary.unsupported_reason)
            continue
        groups.setdefault(summary.batch_consistency_id, []).append(summary)

    all_consistent = True
    failures: Dict[str, List[str]] = {}
    for group_id, members in groups.items():
        if len(members) < 2:
            continue
        lengths = {len(m.sample_hashes) for m in members}
        if len(lengths) > 1:
            all_consistent = False
            failures[group_id] = [
                f"sample count mismatch: {sorted((m.testcase_name, len(m.sample_hashes)) for m in members)}"
            ]
            continue
        for pos in range(next(iter(lengths))):
            distinct = {m.sample_hashes[pos] for m in members}
            if len(distinct) > 1:
                all_consistent = False
                failures.setdefault(group_id, []).append(
                    f"sample[{pos}] differs across {sorted(m.testcase_name for m in members)}"
                )
    return all_consistent, failures
