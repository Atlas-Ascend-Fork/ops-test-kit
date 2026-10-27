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
- worker 比对后：按声明的 relation 样本切片切 output 0——每个样本把
  batch_slice_info 的 (start, stop, step) 直接落在 batch_axis 声明的轴上，
  逐样本 MD5，并以 (各轴 seed 元组, 各轴切片长度元组) 为样本 key 随
  ReturnStructure 回传（主进程内存 O(用例×样本×32B)，替代旧"全量
  output_bytes 回传驻留"）。未声明切片位置的输出不参与；切片粒度（整
  batch / 单 token / 多轴）由声明区间本身决定，框架不做任何布局感知。
- 主进程 cleanup：compare_batch_consistency 按 (api_name, seed, 片段长度)
  跨用例分组 → 同 key 片段 hash 集合基数判定。key 不含声明顺序/位置，
  因此同 seed 样本跨用例换序（reorder）也能对齐比较；api_name 隔离多
  API 混跑时的串比。每个用例声明了几个样本，就参与几次 key 组比对。

接线现状：四通路 worker 侧 assemble（geir/kernel/aclnn/e2e）+ 主进程
InstanceBase.cleanup 侧组装比对。
"""

__all__ = ["slice_batch_hashes", "assemble_batch_summary", "compare_batch_consistency", "BatchSummary", "SampleHash"]

# Standard Packages
import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..testcase.batch import parse_batch_relations

_SLICE_UNSET = "UNSLICEABLE"
_MISSING_META = "BATCH_META_MISSING"
_HASH_LEN = 32  # md5 hexdigest 截断长度（每样本 32B 十六进制）


@dataclass
class SampleHash:
    """单样本输出哈希：seeds/lengths 逐声明轴对齐，digest 为切片字节 MD5。"""

    seeds: Tuple[int, ...]
    lengths: Tuple[int, ...]
    digest: str

    @property
    def key(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """样本分组 key：seed+切片长度（刻意不含声明顺序/位置）。"""
        return self.seeds, self.lengths


@dataclass
class BatchSummary:
    """单用例的 batch 摘要（随 ReturnStructure 回传；主进程内存受控）。"""

    testcase_name: str
    batch_consistency_id: str
    sample_hashes: Optional[List[SampleHash]] = None  # None = 不支持切片（记录原因）
    unsupported_reason: Optional[str] = None
    batch_axis: Optional[Any] = None
    batch_slice_info: Optional[Any] = None
    api_name: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "testcase": self.testcase_name,
            "api_name": self.api_name,
            "sample_hashes": (
                None
                if self.sample_hashes is None
                else [
                    {"seeds": sample.seeds, "lengths": sample.lengths, "digest": sample.digest}
                    for sample in self.sample_hashes
                ]
            ),
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
    batch_fields: Optional[Tuple[Any, Any, Any]],
) -> BatchSummary:
    """worker 比对后切片哈希：按声明的样本切片切 output 0，逐样本 MD5。

    batch_fields = (batch_axis, batch_slice_info, batch_seed) 三字段；解析
    失败 → unsupported（MISSING）。每个声明样本把 (start, stop, step) 直接
    落在 batch_axis 声明的轴上提取片段，产出一个 SampleHash（key=(各轴
    seed, 各轴切片长度)）。只切声明位置——"每个用例替换了几个切片，就
    hash 几次"，未声明位置不参与。样本切片非法/空则 warning 跳过；
    维度不足记 unsupported_reason，不抛——batch 一致性是辅助校验，
    不阻断主流程。batch_consistency_id 仅存于 summary 供对账，不影响哈希。
    """
    summary = BatchSummary(
        testcase_name=testcase_name,
        batch_consistency_id=batch_consistency_id,
    )
    if not isinstance(batch_fields, (tuple, list)) or len(batch_fields) != 3:
        summary.unsupported_reason = _MISSING_META
        return summary
    batch_axis, batch_slice_info, batch_seed = batch_fields
    summary.batch_axis = batch_axis
    summary.batch_slice_info = batch_slice_info
    relations = parse_batch_relations(batch_axis, batch_slice_info, batch_seed, slot_idx=0)
    if relations is None:
        summary.unsupported_reason = _MISSING_META
        return summary
    axes, samples = relations
    if not samples:
        summary.unsupported_reason = _MISSING_META
        return summary
    output = _normalize_output(output)
    if output is None:
        summary.unsupported_reason = _SLICE_UNSET
        return summary
    if output.ndim <= max(axes):
        summary.unsupported_reason = _SLICE_UNSET
        return summary
    summary.sample_hashes = _hash_samples(output, axes, samples, testcase_name)
    return summary


def _normalize_output(output):
    """output → np.ndarray（torch 视图等经 asarray 归一）；失败返回 None。"""
    if isinstance(output, np.ndarray):
        return output
    try:
        return np.asarray(output)
    except Exception:  # noqa: BLE102 归一尽力而为
        return None


def _hash_samples(output, axes, samples, testcase_name):
    """逐样本提取片段并 MD5：非法/空片段 warning 跳过，返回 SampleHash 列表。"""
    hashes = []
    for sample_slices, sample_seeds in samples:
        try:
            value, lengths = _extract_sample(output, axes, sample_slices)
        except Exception as exc:  # noqa: BLE102 切片尽力而为
            logging.warning("[%s] batch consistency skip sample slice %r: %s", testcase_name, sample_slices, exc)
            continue
        if value is None:
            logging.warning(
                "[%s] batch consistency skip sample: invalid slice %r on axes %r",
                testcase_name,
                sample_slices,
                axes,
            )
            continue
        if value.size == 0:
            logging.warning("[%s] batch consistency skip empty sample slice %r", testcase_name, sample_slices)
            continue
        digest = hashlib.md5(np.ascontiguousarray(value).tobytes()).hexdigest()[:_HASH_LEN]  # noqa: S324
        hashes.append(SampleHash(seeds=tuple(sample_seeds), lengths=tuple(lengths), digest=digest))
    return hashes


def _extract_sample(output, axes, sample_slices):
    """单样本片段提取：返回 (value, lengths) 或 (None, None)。

    各声明轴的 (start, stop, step) 直接作用在张量轴上（无布局感知）；
    lengths 与声明轴逐轴对齐。
    """
    indexer = [slice(None)] * output.ndim
    lengths = []
    for axis, (start, stop, step) in zip(axes, sample_slices):
        if step <= 0:
            return None, None
        length = len(range(start, stop, step))
        if length <= 0:
            return None, None
        indexer[axis] = slice(start, stop, step)
        lengths.append(length)
    return output[tuple(indexer)], lengths


def assemble_batch_summary(
    outputs: Sequence,
    testcase,
    batch_consistency_id: Optional[str],
) -> Optional[BatchSummary]:
    """result construct 的装配入口：id 为 None（不参与分组）或输出空 → None。

    relation 三字段任一缺失 → 产出带 unsupported_reason 的 summary（参与
    回传但比对时跳过），而非静默 None——与旧契约一致。
    """
    if batch_consistency_id is None or not outputs:
        return None
    output = outputs[0]
    if isinstance(output, (str, bytes)) or output is None:
        return None
    summary = slice_batch_hashes(
        output,
        getattr(testcase, "testcase_name", None),
        batch_consistency_id,
        (
            getattr(testcase, "batch_axis", None),
            getattr(testcase, "batch_slice_info", None),
            getattr(testcase, "batch_seed", None),
        ),
    )
    summary.api_name = getattr(testcase, "api_name", None) or getattr(testcase, "op_name", None)
    return summary


def _key_repr(api_name, key) -> str:
    seeds, lengths = key
    return f"api={api_name},seed={seeds},slice_len={lengths}"


def _mismatch_detail(entries) -> str:
    """不一致组详情：按 digest 分簇，多数派为基准，列出各簇用例名与摘要前缀。"""
    clusters = defaultdict(list)
    for name, digest in entries:
        clusters[digest].append(name)
    if len(clusters) < 2:
        return ""
    ordered = sorted(clusters.items(), key=lambda item: (-len(item[1]), item[0]))
    parts = [f"{sorted(names)} (digest {digest[:12]}…)" for digest, names in ordered]
    return " vs ".join(parts)


def compare_batch_consistency(summaries: Dict[str, BatchSummary]) -> Tuple[bool, Dict[str, List[str]], int, set]:
    """cleanup 组装比对：按 (api_name, seed, 切片长度) 跨用例分组 → 同 key hash 集合基数判定。

    summaries: {testcase_name: BatchSummary}。判定：每个 key 组内（跨用例、
    跨声明顺序/位置），片段 hash 集合基数 == 1 → 一致；单成员组跳过。
    同用例内同 key 的多个样本也同组互比（同 seed 同长度 = 同关系片段）。
    api_name 隔离不同 API 的样本（多 API 混在一个 CSV 时不串比）；同 API
    的 reorder 对仍按 seed+长度对齐。
    返回 (all_consistent, {key_repr: 不一致描述}, 实际比对的 key 组数,
    不一致组涉及的用例名集合)；unsupported 用例跳过并 log。
    """
    groups: Dict[tuple, List[Tuple[str, str]]] = defaultdict(list)
    for summary in summaries.values():
        if summary is None:
            continue
        if summary.sample_hashes is None:
            logging.info("batch consistency skip [%s]: %s", summary.testcase_name, summary.unsupported_reason)
            continue
        for sample in summary.sample_hashes:
            groups[(summary.api_name,) + sample.key].append((summary.testcase_name, sample.digest))

    all_consistent = True
    failures: Dict[str, List[str]] = {}
    compared_groups = 0
    failed_cases = set()
    for key, entries in groups.items():
        if len(entries) < 2:
            continue
        compared_groups += 1
        distinct = {digest for _, digest in entries}
        if len(distinct) > 1:
            all_consistent = False
            failures[_key_repr(key[0], key[1:])] = [f"sample mismatch: {_mismatch_detail(entries)}"]
            failed_cases.update(name for name, _ in entries)
    return all_consistent, failures, compared_groups, failed_cases
