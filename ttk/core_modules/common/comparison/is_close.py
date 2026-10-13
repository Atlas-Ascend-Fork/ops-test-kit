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
is_close comparison
"""

# Standard Packages
import numpy as np

# Third-party Packages
from ....utilities.dtypes import dtype_to_name
from .registry import (
    CHUNK_SIZE,
    FAIL_REASONS,
    ComparisonBase,
    EachCompareResult,
    _FirstK,
    register_comparison,
)


@register_comparison(["close", "isclose"])
class CloseComparison(ComparisonBase):
    STANDARD_NAME = "isclose"

    # 默认 rtol/ptol 按 dtype 分档：低精度浮点的相邻可表示值相对距离远大于 1e-4
    # （fp8 家族 ulp ≈ 2^-2~2^-3），不分档则非同值对必挂。与 mix_tolerance 表同源。
    # rtol 与 ptol 共用一张分档表（现状语义：两容差的分档默认相同）。
    _DEFAULT_RTOL = {
        "float16": 0.001,
        "bfloat16": 0.001,
        "complex32": 0.001,
        "float8_e4m3fn": 2**-2,
        "float8_e5m2": 2**-1,
    }
    _DEFAULT_ATOL = 1e-8
    _DEFAULT_PTOL = _DEFAULT_RTOL

    def _default_rtol(self, dtype):
        return self._DEFAULT_RTOL.get(dtype_to_name(dtype), 0.0001)

    def _default_ptol(self, dtype):
        return self._DEFAULT_PTOL.get(dtype_to_name(dtype), 0.0001)

    @staticmethod
    def _normalize_dtype(arr: np.ndarray) -> np.ndarray:
        if hasattr(arr, "dtype") and hasattr(arr.dtype, "name"):
            if arr.dtype.name == "bfloat16":
                return arr.astype("float32", copy=False)
            if arr.dtype.name == "int4":
                return arr.astype("int8", copy=False)
        return arr

    @staticmethod
    def _close_mask(out_chunk: np.ndarray, gold_chunk: np.ndarray, rtol: float, atol: float) -> np.ndarray:
        """与 np.isclose(..., equal_nan=True) 逐元素等价，但把窄浮点一次性抬到 float32 再算。

        np.isclose 内部要做 isfinite(x)/isfinite(y)/abs(x-y)/rtol*abs(y)/加法/比较 等约 8 趟运算，
        **每一趟都会独立地把 fp16 转一次 float32**——numpy 对 fp16 没有原生 SIMD 循环，转换是
        逐元素标量完成的。实测同规模 fp16 比 fp32 慢 13 倍（1 亿元素 2.08s vs 0.16s），10.7 亿
        元素一次 isclose 要 110s。这里每块只转一次，后续算术全在 float32 上走 SIMD。
        """
        a = out_chunk.astype(np.float32, copy=False) if out_chunk.dtype == np.float16 else out_chunk
        b = gold_chunk.astype(np.float32, copy=False) if gold_chunk.dtype == np.float16 else gold_chunk
        finite = np.isfinite(a) & np.isfinite(b)
        # 有限位按容差判；非有限位按 np.isclose 的语义走 ==（同号 inf 相等），
        # 再叠加 equal_nan=True 的 both-nan 视为相等。
        within = np.abs(a - b) <= atol + rtol * np.abs(b)
        nonfinite_eq = (a == b) | (np.isnan(a) & np.isnan(b))
        return np.where(finite, within, nonfinite_eq)

    def _isclose(self) -> EachCompareResult:
        # 空数组与元素数不一致已在 base compare() 的 _precheck 统一短路，
        # 这里只处理非空同数：分块扫描：逐块 isclose + 失败计数（precision 精确）
        # + 前 K 个失败位（索引升序，仅 _log_diff_output 消费）
        rtol = self.rtol
        atol = self.atol
        output_size = self.output.size
        golden_size = self.golden.size
        n_bad = 0
        first_k = _FirstK(self.MAX_DIFF_OUTPUT)
        for s in range(0, output_size, CHUNK_SIZE):
            output = self._normalize_dtype(self.output[s : s + CHUNK_SIZE])
            golden = self._normalize_dtype(self.golden[s : s + CHUNK_SIZE])
            if rtol == 0 and atol == 0:
                diff_results = (output == golden) | (np.isnan(output) & np.isnan(golden))
            else:
                diff_results = self._close_mask(output, golden, rtol, atol)
            bad = np.flatnonzero(~diff_results)
            n_bad += int(bad.size)
            first_k.add(bad + s)
        precision = (golden_size - n_bad) / golden_size
        return EachCompareResult(precision, first_k.result(), False, "")

    def compare_impl(self) -> EachCompareResult:
        ptol = self.ptol
        compare_result = self._isclose()
        compare_result.is_pass = not (
            isinstance(compare_result.precision, str) or (1 - compare_result.precision) > ptol
        )
        compare_result.standard = self.STANDARD_NAME
        metrics = {
            "standard": self.STANDARD_NAME,
            "precision": (
                f"{compare_result.precision * 100}%"
                if not isinstance(compare_result.precision, str)
                else compare_result.precision
            ),
            "pass": bool(compare_result.is_pass),
        }
        if not compare_result.is_pass:
            metrics["reason"] = FAIL_REASONS["tolerance_exceeded"]
        compare_result.metrics = metrics
        return compare_result
