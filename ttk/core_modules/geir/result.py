#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""geir/result.py（换基 ReturnStructureBase + dataclass → __slots__ 统一）。"""

from typing import Any

from ..common.result import ReturnStructureBase


class GeirReturnStructure(ReturnStructureBase):
    """GEIR 通路结果结构（cst_/dyn_ 槽 + is_binary 列名切换）。"""

    __slots__ = (
        "precision",
        "passed",
        "precision_status",
        "log",
        "cst_precision",
        "dyn_precision",
        "cst_perf_us",
        "dyn_perf_us",
        "xpu_metrics",
        "precision_metrics",
        "deterministic_status",
        "batch_consistency_id",
    )

    def __init__(self):
        self.precision = ""
        self.passed = False
        self.precision_status = "UNKNOWN"
        self.log = ""
        self.cst_precision = ""
        self.dyn_precision = ""
        self.cst_perf_us = None
        self.dyn_perf_us = None
        self.xpu_metrics = {}
        self.precision_metrics = {}
        self.deterministic_status = None
        self.batch_consistency_id = None
        # dl=3 batch 一致性：worker 切片哈希摘要（随 pipe 回传主进程，非 CSV 列——
        # cleanup 组装比对消费）
        self.batch_summary = None

    @classmethod
    def get_titles(cls, is_binary: bool = False, **kwargs) -> tuple:
        """结果 CSV 列名（is_binary 切换 cst_bin_ ↔ cst_）。"""
        cst_perf = "cst_bin_perf_us" if is_binary else "cst_perf_us"
        dyn_perf = "dyn_bin_perf_us" if is_binary else "dyn_perf_us"
        cst_prec = "cst_bin_precision" if is_binary else "cst_precision"
        dyn_prec = "dyn_bin_precision" if is_binary else "dyn_precision"
        return (
            "precision",
            "precision_status",
            cst_perf,
            dyn_perf,
            cst_prec,
            dyn_prec,
            "xpu_metrics",
            "precision_metrics",
            "deterministic_status",
            "batch_consistency_id",
            "batch_summary",
            "log",
        )

    def construct(self, testcase, compare_result: Any, **kwargs) -> None:
        """聚合抄写——GEIR 为直赋值式（_geir_run 逐模式写槽），无集中 construct。"""

    @staticmethod
    def get_title_indices():
        titles = GeirReturnStructure.get_titles()
        return {t: i for i, t in enumerate(titles)}

    def pick_data(self, case_result_title: tuple) -> tuple:
        data = []
        for title in case_result_title:
            if title == "precision":
                data.append(self.precision)
            elif title == "precision_status":
                data.append(self.precision_status)
            elif title in ("cst_perf_us", "cst_bin_perf_us"):
                data.append(self.cst_perf_us if self.cst_perf_us is not None else "CST_OFF")
            elif title in ("dyn_perf_us", "dyn_bin_perf_us"):
                data.append(self.dyn_perf_us if self.dyn_perf_us is not None else "DYN_OFF")
            elif title in ("cst_precision", "cst_bin_precision"):
                data.append(self.cst_precision)
            elif title in ("dyn_precision", "dyn_bin_precision"):
                data.append(self.dyn_precision)
            elif title == "xpu_metrics":
                data.append(self.xpu_metrics)
            elif title == "precision_metrics":
                data.append(self.precision_metrics)
            elif title == "deterministic_status":
                data.append(self.deterministic_status)
            elif title == "batch_consistency_id":
                data.append(self.batch_consistency_id)
            elif title == "log":
                data.append(self.log)
            else:
                data.append("")
        return tuple(data)
