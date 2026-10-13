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
cosine similarity comparison
"""

# Standard Packages
import numpy as np

# Third-party Packages
from .registry import CHUNK_SIZE, FAIL_REASONS, ComparisonBase, EachCompareResult, register_comparison


@register_comparison("cosine")
class CosineSimilarityComparison(ComparisonBase):
    STANDARD_NAME = "cosine"
    _DEFAULT_RTOL = 0.01

    @staticmethod
    def _normalize_dtype(arr: np.ndarray) -> np.ndarray:
        # convert int4 -> int8, bcz np.dot will be wrong with int4
        if hasattr(arr, "dtype") and getattr(arr.dtype, "name", None) == "int4":
            return arr.astype("int8", copy=False)
        return arr

    def compare_impl(self) -> EachCompareResult:
        rtol = self.rtol
        output = self._normalize_dtype(self.output)
        golden = self._normalize_dtype(self.golden)
        # 分块 float64 累加 dot 与两路平方和，norm=sqrt(平方和)：原生 dtype 直算
        # 时整数 dot 按输入 dtype 回绕、窄浮点平方和溢出；分块免全量 float64 物化
        dot = 0.0
        sq_output = 0.0
        sq_golden = 0.0
        for s in range(0, output.size, CHUNK_SIZE):
            _output = output[s : s + CHUNK_SIZE].astype(np.float64, copy=False)
            _golden = golden[s : s + CHUNK_SIZE].astype(np.float64, copy=False)
            dot += float(np.dot(_output, _golden))
            sq_output += float(np.dot(_output, _output))
            sq_golden += float(np.dot(_golden, _golden))
        output_norm = np.sqrt(sq_output)
        golden_norm = np.sqrt(sq_golden)
        denom = output_norm * golden_norm
        # 零范数：双零向量按 precision=1，单侧零按 0
        precision = (1.0 if output_norm == golden_norm else 0.0) if denom == 0 else float(dot / denom)
        is_pass = (1 - precision) <= rtol
        metrics = {"standard": self.STANDARD_NAME, "precision": float(precision), "pass": bool(is_pass)}
        if not is_pass:
            metrics["reason"] = FAIL_REASONS["similarity_below_threshold"]
        return EachCompareResult(precision, is_pass=is_pass, standard=self.STANDARD_NAME, metrics=metrics)
