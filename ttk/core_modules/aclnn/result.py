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
Op Api NPU Profiling Structure
"""

__all__ = ["AclnnReturnStructure", "ApiComparisonResult", "ApiProfilingResult"]


# Standard Packages
from typing import List, Optional, Union

from ttk.switches import get_global_storage

# Third-Party Packages
from ..common.result import ReturnStructureBase
from .testcase import TestcaseAclnn


def _pick_api_avg_us(api_prof, api_name):
    """Single aclnn call host-side avg time (us), excludes runtime APIs."""
    if not isinstance(api_prof, list):
        return None
    for item in api_prof:
        if item.get("API Name") == api_name:
            try:
                return round(float(item.get("Avg(us)", 0)), 3)
            except (TypeError, ValueError):
                return None
    return None


def _sum_op_avg_us(op_prof):
    """Sum of all NPU op kernel avg time (us) per call."""
    if not isinstance(op_prof, list):
        return None
    total = 0.0
    for item in op_prof:
        try:
            total += float(item.get("Avg Time(us)", 0))
        except (TypeError, ValueError):
            return None
    return round(total, 3)


class ApiProfilingResult:
    """
    RTS Profiling output
    """

    def __init__(
        self,
        success: bool,
        api_prof=None,
        op_prof=None,
        output_bytes=(None,),
        output_view_shapes=(None,),
        oob: str = "UNKNOWN",
        deterministic_status: Optional[str] = None,
        npu_memory: Optional[float] = None,
        execute_error: Optional[str] = None,
    ):
        self.api_prof: Union[str, List[dict]] = api_prof
        self.op_prof: Union[str, List[dict]] = op_prof
        self.output_bytes: Optional[Union[tuple, list]] = output_bytes
        self.output_view_shapes: Optional[Union[tuple, list]] = output_view_shapes
        self.oob: Optional[str] = oob
        self.success = success
        self.deterministic_status: Optional[str] = deterministic_status
        # NPU device workspace memory (MB) requested via aclnn GetWorkspaceSize.
        self.npu_memory: Optional[float] = npu_memory
        # Execute failure detail (plog errors / exception text), e.g. tiling interception.
        self.execute_error: Optional[str] = execute_error

    @classmethod
    def fail(cls, fail_result: str) -> "ApiProfilingResult":
        return cls(False, fail_result, fail_result, (fail_result,), ("NO_OUTPUT",), "UNKNOWN")

    @property
    def oob_status(self):
        if not self.oob:
            return "PASS"
        oob_lst = self.oob.split(",")
        return "FAIL" if "FAIL" in oob_lst else "PASS"

    def failed(self):
        return not self.success and self.api_prof != "SUPPRESSED"


class ApiComparisonResult:
    __slots__ = ("precision", "passed", "metrics")

    def __init__(self, default_value):
        self.precision = default_value
        self.passed = default_value
        self.metrics = {}

    def set(self, a, b, metrics=None):
        self.precision = a
        self.passed = b
        self.metrics = metrics or {}
        return self

    def get(self) -> tuple:
        return tuple(getattr(self, name) for name in self.__slots__)


class AclnnReturnStructure(ReturnStructureBase):
    """
    Structure for op api profiling return content.
    """

    __slots__ = (
        "precision",
        "precision_status",
        "precision_metrics",
        "batch_consistency_id",
        "deterministic_status",
        "soc",
        "api_perf_us",
        "op_perf_us",
        "perf_status",
        "xpu_metrics",
        "npu_memory",
        "golden_device_perf_us",
        "golden_kernel_count",
        "golden_kernel_details",
        "golden_source",
    )

    EXTRA_SLOTS = (
        "precision",
        "soc",
        "api_perf_us",
        "op_perf_us",
        "perf_status",
    )

    def __init__(self, default_value=None):
        self.precision = default_value
        # Precision
        self.precision_status = default_value
        self.precision_metrics = default_value
        # Special
        self.batch_consistency_id = default_value
        self.deterministic_status = default_value
        self.soc = get_global_storage().dev_plat
        # Performance
        self.api_perf_us = default_value
        self.op_perf_us = default_value
        self.perf_status = default_value
        self.xpu_metrics = {}
        # NPU device workspace memory (MB) from aclnn GetWorkspaceSize.
        self.npu_memory = default_value
        # golden-mode NPU：golden 执行侧性能（设备阶段采集；CPU/XPU 路径为 None）
        self.golden_device_perf_us = default_value
        self.golden_kernel_count = default_value
        self.golden_kernel_details = default_value
        self.golden_source = default_value
        # dl=3 batch 一致性：worker 切片哈希摘要（随 pipe 回传主进程，非 CSV 列——
        # cleanup 组装比对消费）
        self.batch_summary = None

    # noinspection DuplicatedCode
    def construct(self, context: TestcaseAclnn, compare_result: ApiComparisonResult):
        """Construct the structure with context"""
        # Check prof_results and construct one if necessary
        self.precision = compare_result.precision
        self.precision_status = compare_result.passed
        self.precision_metrics = compare_result.metrics or {}
        self.batch_consistency_id = getattr(context, "batch_consistency_id", None)
        self.xpu_metrics = getattr(context, "xpu_metrics", {})
        # Performance data from msprof (already collected in ApiProfilingResult)
        prof_result = getattr(context, "prof_result", None)
        api_prof = getattr(prof_result, "api_prof", None) if prof_result else None
        op_prof = getattr(prof_result, "op_prof", None) if prof_result else None
        self.api_perf_us = _pick_api_avg_us(api_prof, getattr(context, "api_name", None))
        self.op_perf_us = _sum_op_avg_us(op_prof)
        self.perf_status = "PASS" if self.op_perf_us is not None else "UNKNOWN"
        self.npu_memory = getattr(prof_result, "npu_memory", None) if prof_result else None
        from ..common.golden import golden_perf_columns

        for key, value in golden_perf_columns(
            getattr(context, "golden_perf", None), getattr(context, "golden_source", None)
        ).items():
            setattr(self, key, value)

    @classmethod
    def get_titles(cls, **kwargs) -> tuple:
        return cls.__slots__
