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
re-quantize comparison

fp8 家族 + hifloat8：view 成 int8 按 ULP 距离比对（|位模式差| <= 1）。
其余 dtype（CLI 显式路由到本判据的 fp16/fp32/bf16 等）：按 |值差| <= 1 比对
——与 ULP 语义不同，是绝对值容差。
"""

# Standard Packages
import numpy as np

# Third-party Packages
from .registry import (
    CHUNK_SIZE,
    FAIL_REASONS,
    ComparisonBase,
    EachCompareResult,
    _FirstK,
    register_comparison,
)


@register_comparison("requant")
class ReQuantizeComparison(ComparisonBase):
    STANDARD_NAME = "requant"
    _DEFAULT_PTOL = 0

    # view 成 int8 做 ULP 比对的 dtype（fp8 家族 + hifloat8）
    _ULP_VIEW_DTYPES = ("float8_e5m2", "float8_e4m3fn", "hifloat8")

    def _is_ulp_view(self):
        return str(self.output.dtype).split(".")[-1] in self._ULP_VIEW_DTYPES

    def _default_ptol(self, dtype):
        # fp8 家族默认放宽 0.1%，Spec.tolerance 可配 ptol 覆盖
        return 0.001 if str(dtype).split(".")[-1] in self._ULP_VIEW_DTYPES else 0

    def compare_impl(self) -> EachCompareResult:
        # 分块扫描。ULP 差值不能在 int8 上做 np.subtract：int8 视图两端值差可达
        # ±255（如 127-(-128)），原生减法回绕成 -1 → |diff|=1 ≤ 1 → 255 ULP 的
        # 偏差误判 PASS。提升 int16 再减（差值域 ±255 完整覆盖）。
        view_ulp = self._is_ulp_view()
        ptol = self.ptol
        golden_size = self.golden.size
        diff_size = 0
        first_k = _FirstK(self.MAX_DIFF_OUTPUT)
        for s in range(0, self.output.size, CHUNK_SIZE):
            output = self.output[s : s + CHUNK_SIZE]
            golden = self.golden[s : s + CHUNK_SIZE]
            if view_ulp:
                output = output.view(np.int8)
                golden = golden.view(np.int8)
                diff_results = np.abs(np.subtract(output, golden, dtype=np.int16))
            elif np.issubdtype(output.dtype, np.integer):
                # 整型差值回绕防护：同上，整型统一提升 int64 再减（int 输入经
                # Spec quant 声明 + CLI --compare requant 可路由到本判据）。
                diff_results = np.abs(np.subtract(output, golden, dtype=np.int64))
            else:
                diff_results = np.abs(np.subtract(output, golden))
            bad = np.flatnonzero(diff_results > 1)
            # NaN 位置双方都是 NaN 时视为相等（排除出失败位）
            both_nan = np.isnan(self.output[s : s + CHUNK_SIZE]) & np.isnan(self.golden[s : s + CHUNK_SIZE])
            if bad.size and both_nan.any():
                bad = bad[~both_nan[bad]]
            diff_size += int(bad.size)
            first_k.add(bad + s)

        precision = (golden_size - diff_size) / golden_size
        is_pass = (1 - precision) <= ptol
        metrics = {"standard": self.STANDARD_NAME, "precision": f"{precision * 100}%", "pass": bool(is_pass)}
        if not is_pass:
            metrics["reason"] = FAIL_REASONS["precision_exceeded"]
        return EachCompareResult(
            precision,
            first_k.result(),
            is_pass=is_pass,
            standard=self.STANDARD_NAME,
            metrics=metrics,
        )
