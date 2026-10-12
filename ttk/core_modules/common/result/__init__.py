#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""common/result：worker 结果的聚合点与产物契约。

- CompareResult：比对产物的纯数据类型（comparison 产出、result 只存）。
- ReturnStructureBase：worker 结果打包结构契约。可 pickle（forkserver pipe
  回传前置约束）；construct 是聚合抄写（只存结果不调比对）；签名不统一是
  有意设计（通路数据来源本质不同，强行统一只会造 out 参数）。

公共列（COMMON_SLOTS：precision_status/precision_metrics/
batch_consistency_id/deterministic_status/xpu_metrics/npu_memory）由
各通路结果结构体继承声明。
"""

__all__ = ["ReturnStructureBase", "CompareResult", "format_xpu_metrics"]

# Standard Packages
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class CompareResult:
    """比对产物的纯数据类型（comparison 包产出，result 只存不调）。

    precision：精度值（"0.9987" 或占位 "COMPARE_FAILURE"）；
    passed："PASS" / "FAIL" / 占位；metrics：判据量化指标。
    """

    precision: Any = None
    passed: Any = None
    metrics: dict = None


class ReturnStructureBase(ABC):
    """worker 结果打包结构契约（聚合抄写，可 pickle）。

    子类要求：
    1. 类属性 COMMON_SLOTS/EXTRA_SLOTS 声明列；get_titles 由其生成
    2. construct(testcase, compare_result, **kwargs) 聚合抄写
    3. pick_data(titles) 按 titles 取行 tuple；缺失列填 None
    禁止：调比对、调 profiler、改 testcase 声明态。
    """

    COMMON_SLOTS = (
        "precision_status",
        "precision_metrics",
        "batch_consistency_id",
        "deterministic_status",
        "xpu_metrics",
        "npu_memory",
    )
    EXTRA_SLOTS = ()

    @classmethod
    @abstractmethod
    def get_titles(cls, **kwargs) -> tuple:
        """结果 CSV 列名（表头与 pick_data 的取数序）= 公共列 + 专属列。"""

    @abstractmethod
    def construct(self, testcase, compare_result: Any, **kwargs) -> None:
        """聚合抄写——契约只约定"结束时结构完整、可 pick_data"。"""

    def pick_data(self, titles: tuple) -> tuple:
        """按 titles 取行 tuple；缺失列填 None。"""
        return tuple(getattr(self, title, None) for title in titles)


def format_xpu_metrics(xpu_results):
    """Format xpu_results dict into xpu_metrics dict for CSV output."""
    if not xpu_results:
        return {}
    metrics = {}
    for provider, entry in xpu_results.items():
        m = {"status": entry.get("status", "FAIL"), "api": entry.get("api", "")}
        if entry.get("perf"):
            m["device_us"] = entry["perf"].get("device_us", "NA")
            m["peak_memory_mb"] = entry["perf"].get("peak_memory_mb", "NA")
        if entry.get("error"):
            m["error"] = entry["error"]
        metrics[provider] = m
    return metrics
