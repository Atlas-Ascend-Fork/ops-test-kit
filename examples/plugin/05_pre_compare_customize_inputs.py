#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Advanced: small-op composition, golden reusing bench, customize_inputs, pre_compare"""

import numpy
import torch


class SoftmaxAdvancedSpec:
    """Small-op composition + golden reusing bench"""

    def golden(x, *, axis=-1, **kwargs):
        """golden reusing bench logic — calls ThirdPartyImpl on CPU tensor"""
        cpu_tensor = torch.from_numpy(x)
        result = SoftmaxAdvancedSpec._BenchImpl()(cpu_tensor, axis=axis)
        return [r.numpy() for r in result]

    class _BenchImpl:
        def __init__(self, *, axis=-1, **kwargs):
            self.axis = axis

        def __call__(self, x, *, axis=None, **kwargs):
            dim = axis if axis is not None else self.axis
            return [torch.nn.functional.softmax(x, dim=dim)]

    third_party = {
        "torch": "torch.nn.functional.softmax",
        "npu_decompose": _BenchImpl,
    }

    tolerance = {
        "float32": {"standard": "mix_tolerance"},
        "float16": {"standard": "mix_tolerance"},
    }


class HistogramInputSpec:
    """customize_inputs — custom input generation"""

    def golden(x, min_val, max_val, **kwargs):
        return [numpy.histogram(x, bins=10, range=(min_val[0], max_val[0]))[0]]

    def customize_inputs(x, min_val, max_val, **kwargs):
        """Parameter names match operator definition, returns modified input arrays"""
        min_data = min_val[0]
        max_data = max_val[0]
        if min_data == max_data:
            min_val[0] = numpy.min(x)
            max_val[0] = numpy.max(x)
        return (x, min_val, max_val)


class TopKSpec:
    """pre_compare (in-place mode) + compare.

    pre_compare mode 1 — in-place (no return):
        Modify arrays via [:], return None. Requires shape unchanged.
        Framework sees the modified arrays directly.
    """

    def golden(x, *, k, **kwargs):
        return [numpy.sort(x.flatten())[-k:]]

    def pre_compare(*outputs, **kwargs):
        """In-place sort: outputs[0]=NPU, outputs[1]=golden. Returns None."""
        outputs[0][:] = numpy.sort(outputs[0])
        outputs[1][:] = numpy.sort(outputs[1])

    def compare(*outputs, **kwargs):
        """Custom compare: exact match with diff count.

        Return dict (single output) or list[dict] (multi output).
        Each dict fields:
          pass         (bool, required)   — per-output pass/fail
          precision    (str|float, required) — float is percentage (99.98, not 0.9998)
          error_info   (str, optional)    — error description for debug log
          metrics      (dict, optional)   — structured metrics for future reporting/aggregation,
                                             e.g. {"max_abs_diff": 0.001, "cosine_sim": 0.9998}.
                                             Framework does not consume currently.
          diff_indices (list, optional)   — mismatch indices, 1D post-flatten
        """
        npu_out, golden_out = outputs[0], outputs[1]
        npu_flat = npu_out.flatten()
        golden_flat = golden_out.flatten()
        diff_mask = npu_flat != golden_flat
        diff_idx = numpy.where(diff_mask)[0].tolist()
        precision = (npu_flat.size - len(diff_idx)) / npu_flat.size * 100
        return {
            "pass": len(diff_idx) == 0,
            "precision": precision,
            "diff_indices": diff_idx,
            "error_info": f"{len(diff_idx)} mismatches" if diff_idx else None,
            "metrics": {"max_abs_diff": float(numpy.max(numpy.abs(npu_flat - golden_flat)))},
        }


class TopKReturnSpec:
    """pre_compare (return-value mode) + compare.

    pre_compare mode 2 — return-value:
        Return a list of transformed outputs, framework replaces originals.
        Use this when transform changes shape (e.g. truncation, reshape).
    """

    def golden(x, *, k, **kwargs):
        return [numpy.sort(x.flatten())[-k:]]

    def pre_compare(*outputs, **kwargs):
        """Return-value mode: truncate outputs to first-k elements, then sort.
        k is assumed from the operator attribute (kwargs not yet passed by framework)."""
        k = kwargs.get("k", len(outputs[0]))  # fallback: assume all elements
        truncated = [arr[:k] if len(arr.shape) > 0 else arr for arr in outputs]
        sorted_outs = [numpy.sort(arr) for arr in truncated]
        return sorted_outs

    def compare(*outputs, **kwargs):
        npu_out, golden_out = outputs[0], outputs[1]
        diff = numpy.abs(npu_out.flatten() - golden_out.flatten())
        diff_idx = numpy.where(diff > 1e-6)[0].tolist()
        precision = (npu_out.size - len(diff_idx)) / npu_out.size * 100
        return {
            "pass": len(diff_idx) == 0,
            "precision": precision,
            "diff_indices": diff_idx,
            "metrics": {"max_abs_diff": float(numpy.max(diff)) if diff_idx else 0.0},
        }


class UniqueSpec:
    """pre_compare with tensor-list (nested outputs) — in-place mode.

    When output_dist has tensor-list positions, pre_compare receives nested
    structure: single output → numpy array, tensor-list output → list of arrays.
    In-place mode: modify each array via [:], return None.
    """

    def golden(x, **kwargs):
        return [numpy.sort(x.flatten()), numpy.argsort(x.flatten())]

    def pre_compare(*outputs, **kwargs):
        """In-place sort both outputs (values + indices). outputs[0]=NPU [values, indices]."""
        for output in outputs:
            if isinstance(output, list):
                for arr in output:
                    arr[:] = numpy.sort(arr)
            else:
                output[:] = numpy.sort(output)

    def compare(*outputs, **kwargs):
        """Multi-output compare: return list[dict], one per output."""
        results = []
        for npu_out, golden_out in zip(outputs[0::2], outputs[1::2]):
            if isinstance(npu_out, list):
                for n, g in zip(npu_out, golden_out):
                    diff = numpy.abs(n.flatten() - g.flatten())
                    diff_idx = numpy.where(diff > 1e-6)[0].tolist()
                    precision = (n.size - len(diff_idx)) / n.size * 100
                    results.append(
                        {
                            "pass": len(diff_idx) == 0,
                            "precision": precision,
                            "diff_indices": diff_idx,
                            "metrics": {"max_abs_diff": float(numpy.max(diff)) if diff_idx else 0.0},
                        }
                    )
            else:
                diff = numpy.abs(npu_out.flatten() - golden_out.flatten())
                diff_idx = numpy.where(diff > 1e-6)[0].tolist()
                precision = (npu_out.size - len(diff_idx)) / npu_out.size * 100
                results.append(
                    {
                        "pass": len(diff_idx) == 0,
                        "precision": precision,
                        "diff_indices": diff_idx,
                        "metrics": {"max_abs_diff": float(numpy.max(diff)) if diff_idx else 0.0},
                    }
                )
        return results


# Explicit registration: class names use *Spec suffix (not *TestSpec),
# so __spec__ dict is needed for discovery.
__spec__ = {
    "softmax_advanced": "SoftmaxAdvancedSpec",
    "histogram_input": "HistogramInputSpec",
    "topk": "TopKSpec",
    "topk_return": "TopKReturnSpec",
    "unique": "UniqueSpec",
}
