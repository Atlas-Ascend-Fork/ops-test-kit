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
Result data structures for framework_api profiling.

FrameworkApiReturnStructure 聚合单用例各形态（eager/graph×3）的精度与性能
结果；pick_data 按 get_titles 列序取数写结果 CSV，列序锚定 __slots__ 原序。
"""

import json

from ..common.result import ReturnStructureBase


class FrameworkApiReturnStructure(ReturnStructureBase):
    """Return structure from worker run, written to result CSV."""

    __slots__ = (
        "precision_status",
        "eager_precision",
        "eager_device_perf_us",
        "eager_cpu_perf_us",
        "eager_kernel_count",
        "eager_kernel_details",
        "graph_cst_precision",
        "graph_cst_device_perf_us",
        "graph_cst_cpu_perf_us",
        "graph_cst_kernel_count",
        "graph_cst_kernel_details",
        "graph_dyn_precision",
        "graph_dyn_device_perf_us",
        "graph_dyn_cpu_perf_us",
        "graph_dyn_kernel_count",
        "graph_dyn_kernel_details",
        "graph_aclgraph_precision",
        "graph_aclgraph_device_perf_us",
        "graph_aclgraph_cpu_perf_us",
        "graph_aclgraph_kernel_count",
        "graph_aclgraph_kernel_details",
        "precision_metrics",
        "batch_consistency_id",
        "deterministic_status",
        "xpu_metrics",
        "npu_memory",
    )

    # 专属列 = __slots__ 去公共槽（COMMON_SLOTS 由基类声明）派生——单源维护，
    # 与 __slots__ 原序一致，结果 CSV 列序不变
    EXTRA_SLOTS = tuple(s for s in __slots__ if s not in ReturnStructureBase.COMMON_SLOTS)

    def __init__(self):
        self.precision_status = None
        self.eager_precision = None
        self.eager_device_perf_us = None
        self.eager_cpu_perf_us = None
        self.eager_kernel_count = None
        self.eager_kernel_details = None
        self.graph_cst_precision = None
        self.graph_cst_device_perf_us = None
        self.graph_cst_cpu_perf_us = None
        self.graph_cst_kernel_count = None
        self.graph_cst_kernel_details = None
        self.graph_dyn_precision = None
        self.graph_dyn_device_perf_us = None
        self.graph_dyn_cpu_perf_us = None
        self.graph_dyn_kernel_count = None
        self.graph_dyn_kernel_details = None
        self.graph_aclgraph_precision = None
        self.graph_aclgraph_device_perf_us = None
        self.graph_aclgraph_cpu_perf_us = None
        self.graph_aclgraph_kernel_count = None
        self.graph_aclgraph_kernel_details = None
        self.batch_consistency_id = None
        self.deterministic_status = None
        self.precision_metrics = {}
        self.xpu_metrics = {}
        self.npu_memory = None
        # dl=3 batch 一致性：worker 切片哈希摘要（随 pipe 回传主进程，非 CSV 列——
        # cleanup 组装比对消费）
        self.batch_summary = None

    @classmethod
    def get_titles(cls, **kwargs):
        return cls.__slots__

    _MODE_PREFIX = {"static": "cst", "dynamic": "dyn", "aclgraph": "aclgraph"}

    def construct(self, precision_str, precision_passed, profile_result, mode=None, metrics=None):
        """
        Build from comparison and profiling results.

        Args:
            precision_str: precision value string from comparison
            precision_passed: "PASS" or "FAIL"
            profile_result: ProfileResult from profiler (may be None)
            mode: None for eager, "static" or "dynamic" for graph
            metrics: per-mode precision metrics dict (accumulated by mode key)
        """
        prefix = "eager_" if mode is None else f"graph_{self._MODE_PREFIX[mode]}_"

        setattr(self, f"{prefix}precision", precision_str)
        if precision_passed == "FAIL" or self.precision_status is None:
            self.precision_status = precision_passed

        if profile_result:
            if profile_result.elapsed_us > 0:
                setattr(self, f"{prefix}device_perf_us", f"{profile_result.elapsed_us:.3f}")
            else:
                setattr(self, f"{prefix}device_perf_us", "----")
            if profile_result.kernel_details:
                kd = profile_result.kernel_details
                setattr(self, f"{prefix}cpu_perf_us", f"{kd.total_cpu_us:.3f}")
                setattr(self, f"{prefix}kernel_count", str(len(kd.kernels)))
                setattr(
                    self,
                    f"{prefix}kernel_details",
                    json.dumps(
                        [
                            {
                                "name": k.name,
                                "total": round(k.device_us, 3),
                                "avg": round(k.avg_us, 3),
                                "max": round(k.max_us, 3),
                                "min": round(k.min_us, 3),
                                "calls": k.calls,
                            }
                            for k in kd.kernels
                        ],
                        ensure_ascii=False,
                    ),
                )

        if metrics:
            mode_key = "eager" if mode is None else f"graph_{self._MODE_PREFIX[mode]}"
            self.precision_metrics[mode_key] = metrics
