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
from .registry import FAIL_REASONS, ComparisonBase, EachCompareResult, register_comparison


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
        common_dtype = np.promote_types(output.dtype, golden.dtype)
        _output = output.astype(common_dtype, copy=False)
        _golden = golden.astype(common_dtype, copy=False)
        # 小动态范围浮点（fp16/bf16/fp8 家族）的 dot/norm 会溢出（如 e5m2 max
        # 57344，百级元素 × 量级 ~30 的平方和即溢出为 inf → precision=inf/nan），
        # 一律提升 fp32 计算，保证余弦相似度精度与输入 dtype 无关
        name = getattr(common_dtype, "name", "")
        compute_dtype = np.float32 if name in ("float16", "bfloat16") or name.startswith("float8") else common_dtype
        _output = _output.astype(compute_dtype, copy=False)
        _golden = _golden.astype(compute_dtype, copy=False)
        output_norm = np.linalg.norm(_output)
        golden_norm = np.linalg.norm(_golden)
        precision = np.dot(_output, _golden.T) / (output_norm * golden_norm)
        is_pass = (1 - precision) <= rtol
        metrics = {"standard": self.STANDARD_NAME, "precision": float(precision), "pass": bool(is_pass)}
        if not is_pass:
            metrics["reason"] = FAIL_REASONS["similarity_below_threshold"]
        return EachCompareResult(precision, is_pass=is_pass, standard=self.STANDARD_NAME, metrics=metrics)
